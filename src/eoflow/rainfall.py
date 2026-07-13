"""
rainfall.py

NIMROD 1 km composite radar rainfall data access for the eoflow package.

This module provides the full NIMROD pipeline:

1. **Downloading** – fetch NIMROD composite ``.tar`` archives from the CEDA
   archive (requires a ``CEDA_TOKEN``).
2. **Cropping** – unpack each archive and crop every 5-minute timestep to a
   study-area shapefile, writing per-timestep NetCDF files to disk.
3. **Loading** – read the per-timestep files back for a given catchment
   polygon and time window, for use by :mod:`eoflow.samples`.

These capabilities are exposed as importable Python functions suitable for
use in notebooks and pipelines, as well as thin CLI wrappers in the
``scripts/`` directory. Per-timestep files can optionally be consolidated
into per-day NetCDFs with ``scripts/consolidate_nimrod.sh`` for faster
repeated loading — see ``docs/PIPELINE.md``.
"""

from __future__ import annotations

import gzip
import os
import shutil
import tarfile
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import iris
import iris.cube
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import requests
import xarray as xr

from eoflow.log_utils import get_logger

logger = get_logger(__name__)

# Silence iris deprecation/future warnings for the whole process.
# iris.FUTURE.save_split_attrs = True
# iris.FUTURE.date_microseconds = True

# HDF5 (the backend for NetCDF4) is not thread-safe by default.
# Serialise all iris.save() calls with this lock so concurrent worker
# threads don't corrupt each other's HDF5 context.
_IRIS_WRITE_LOCK = threading.Lock()


class _NimrodCropCache:
    """Pre-computed spatial crop parameters reused across all timesteps in a batch.

    Avoids re-deriving bounding-box slices and rebuilding the rasterio
    geometry mask for every ``.dat.gz`` file when the NIMROD grid is
    identical across all timesteps (which it always is for a given day).
    """

    __slots__ = ("x_slice", "y_slice", "ndim", "x_dim_first", "mask")

    def __init__(
        self,
        x_slice: slice,
        y_slice: slice,
        ndim: int,
        x_dim_first: bool,
        mask: np.ndarray,
    ) -> None:
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.ndim = ndim
        self.x_dim_first = x_dim_first
        self.mask = mask  # orientation-corrected; ready to apply directly


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def parse_datetime(s: str) -> datetime:
    """Parse a ``YYYY-MM-DDTHH:MM`` or ``YYYY-MM-DD`` string into an aware UTC datetime.

    Parameters
    ----------
    s :
        Date/time string.

    Returns
    -------
    datetime
        UTC-aware datetime.

    Raises
    ------
    ValueError
        If *s* cannot be parsed.
    """
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime '{s}'.  Expected YYYY-MM-DDTHH:MM or YYYY-MM-DD.")


# ---------------------------------------------------------------------------
# Polygon API
# ---------------------------------------------------------------------------


def _polygon_to_bng(polygon):
    """Reproject a WGS-84 Shapely Polygon or MultiPolygon to British National Grid (EPSG:27700).

    Parameters
    ----------
    polygon :
        Input geometry in WGS 84 (EPSG:4326).

    Returns
    -------
    shapely.geometry.Polygon or MultiPolygon
        The reprojected geometry in EPSG:27700 (coordinates in BNG metres).
    """
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    return shapely_transform(transformer.transform, polygon)


# ---------------------------------------------------------------------------
# NIMROD 1km composite - load from disk (public)
# ---------------------------------------------------------------------------


def get_nimrod_rainfall_for_polygon(
    polygon,
    start: str | datetime,
    end: str | datetime,
    *,
    nimrod_dir: Path | str,
    parallel: bool = False,
) -> xr.DataArray:
    """Load NIMROD 1 km composite rainfall data from disk, clipped to a polygon.

    Reads pre-processed NetCDF files written by :func:`process_nimrod_local`
    (or per-day files consolidated by ``scripts/consolidate_nimrod.sh``) and
    returns a rainfall :class:`~xarray.DataArray` masked to the catchment
    polygon, with dimensions ``(time, y, x)``, units ``mm/h``, CRS
    EPSG:27700.

    Parameters
    ----------
    polygon :
        Area of interest in **WGS 84 (EPSG:4326)**.  May be a
        ``shapely.geometry.Polygon`` or ``MultiPolygon``.
    start :
        Start of the valid-time range (UTC, inclusive).  Either a
        ``YYYY-MM-DDTHH:MM`` / ISO-8601 string or a
        :class:`~datetime.datetime`.
    end :
        End of the valid-time range (UTC, inclusive).
    nimrod_dir :
        Path to the NIMROD data on disk.  Any of:

        * A **directory** of per-timestep NetCDF files in the layout
          produced by :func:`process_nimrod_local`::

              <nimrod_dir>/{year}/{YYYYMMDD}/{YYYYMMDD_HHMMSS}.nc

        * A **directory** of per-day NetCDF files produced by
          ``scripts/consolidate_nimrod.sh``::

              <nimrod_dir>/{year}/{YYYYMMDD}.nc

        * A **single file** covering an arbitrary range (e.g. a whole year
          merged with ``ncrcat``).
    parallel :
        If ``True``, open the per-timestep files concurrently via
        ``dask.delayed`` (faster for large time ranges, but may cause
        issues in some multiprocessing environments).  Defaults to
        ``False``.

    Returns
    -------
    xarray.DataArray
        Rainfall rate with dimensions ``(time, y, x)``:

        - ``time``  – UTC timestamps.
        - ``y``     – BNG northings (metres, EPSG:27700), south-to-north.
        - ``x``     – BNG eastings (metres, EPSG:27700), west-to-east.

        Grid cells outside *polygon* are ``NaN``.  Returns an **empty**
        DataArray (shape ``(0, 0, 0)``) when no data are found.

        ``attrs`` on the returned array:

        - ``units``      ``"mm/h"``
        - ``long_name``  ``"Rainfall rate"``
        - ``crs``        ``"EPSG:27700"``
        - ``source``     ``"NIMROD 1km composite"``

    Raises
    ------
    ValueError
        If *end* is before *start*, or if the polygon has zero BNG area.
    """
    import shapely

    if isinstance(start, str):
        start = parse_datetime(start)
    if isinstance(end, str):
        end = parse_datetime(end)

    if end < start:
        raise ValueError("'end' must be >= 'start'")

    nimrod_path = Path(nimrod_dir)

    # --- Reproject polygon to BNG and get bounding box -----------------------
    polygon_bng = _polygon_to_bng(polygon)
    if polygon_bng.area == 0.0:
        raise ValueError("Input polygon has zero area after reprojection to BNG.")

    x_min_bb, y_min_bb, x_max_bb, y_max_bb = polygon_bng.bounds

    logger.info("get_nimrod_rainfall_for_polygon")
    logger.info(
        "  Polygon BNG bounds : %.0f, %.0f \u2192 %.0f, %.0f",
        x_min_bb,
        y_min_bb,
        x_max_bb,
        y_max_bb,
    )
    logger.info("  Time range         : %s \u2192 %s", start.isoformat(), end.isoformat())
    logger.info("  NIMROD source      : %s", nimrod_path)

    _empty_da = xr.DataArray(
        data=np.empty((0, 0, 0), dtype=np.float32),
        dims=["time", "y", "x"],
        name="rainfall_rate",
        attrs={"units": "mm/h", "crs": "EPSG:27700"},
    )

    # BNG projection coordinate names used by Iris-saved NetCDF files
    _X = "projection_x_coordinate"
    _Y = "projection_y_coordinate"

    def _fix_time(ds: xr.Dataset) -> xr.Dataset:
        """Promote a scalar time coordinate to a 1-element dimension.

        Iris saves single-timestep cubes with time as a 0-d (scalar)
        coordinate.  xarray needs it to be a 1-element dimension so that
        ``concat_dim="time"`` can stack multiple files.
        """
        if "time" in ds.coords and ds["time"].ndim == 0:
            return ds.expand_dims("time")
        return ds

    # --- Load from disk -------------------------------------------------------
    try:
        if nimrod_path.is_file():
            # Single consolidated .nc file
            logger.info("Opening consolidated NIMROD file: %s", nimrod_path.name)
            ds = xr.open_dataset(nimrod_path, chunks={"time": 288})
        else:
            # Directory of either per-timestep files (process_nimrod_local) or
            # per-day consolidated files (scripts/consolidate_nimrod.sh).
            # find_nimrod_nc_files expects naive datetimes (file stems have no tz)
            start_naive = start.replace(tzinfo=None)
            end_naive = end.replace(tzinfo=None)
            nc_files, n_skipped = find_nimrod_nc_files(nimrod_path, start_naive, end_naive)
            if not nc_files:
                logger.warning(
                    "No NIMROD NetCDF files found in %s for %s \u2192 %s",
                    nimrod_path,
                    start.isoformat(),
                    end.isoformat(),
                )
                return _empty_da
            logger.info(
                "Found %d file(s) (%d outside requested range), loading ...",
                len(nc_files),
                n_skipped,
            )
            ds = xr.open_mfdataset(
                [str(f) for f in nc_files],
                parallel=parallel,
                preprocess=_fix_time,
                combine="nested",
                concat_dim="time",
                chunks={"time": 288},
                compat="override",
                coords="minimal",
            )

        # Slice to the requested time window. This is required (not just an
        # optimisation): per-day files can only be filtered by
        # find_nimrod_nc_files at day granularity, so a requested window
        # starting or ending mid-day would otherwise pull in whole extra
        # days of data.
        if "time" in ds.coords:
            t0 = np.datetime64(start.replace(tzinfo=None), "ns")
            t1 = np.datetime64(end.replace(tzinfo=None), "ns")
            ds = ds.sel(time=slice(t0, t1))
    except Exception as exc:
        logger.error("Failed to load NIMROD data from %s: %s", nimrod_path, exc, exc_info=True)
        return _empty_da

    # --- Verify expected BNG projection coordinates are present --------------
    for coord in (_X, _Y):
        if coord not in ds.coords:
            logger.error(
                "Expected BNG coordinate '%s' not found in dataset. Available: %s",
                coord,
                list(ds.coords),
            )
            return _empty_da

    # --- Spatial clip to polygon bounding box --------------------------------
    x_vals = ds[_X].values
    y_vals = ds[_Y].values

    x_idx = np.where((x_vals >= x_min_bb) & (x_vals <= x_max_bb))[0]
    y_idx = np.where((y_vals >= y_min_bb) & (y_vals <= y_max_bb))[0]

    if x_idx.size == 0 or y_idx.size == 0:
        logger.warning(
            "Polygon bounds do not intersect the NIMROD grid "
            "(x: %.0f\u2013%.0f m, y: %.0f\u2013%.0f m).",
            x_min_bb,
            x_max_bb,
            y_min_bb,
            y_max_bb,
        )
        return _empty_da

    ds_clip = ds.isel({_X: x_idx, _Y: y_idx})  # type: ignore[arg-type]

    # --- Select the primary data variable ------------------------------------
    data_vars = list(ds_clip.data_vars)
    if not data_vars:
        logger.error("No data variables found in NIMROD dataset.")
        return _empty_da

    var_name = data_vars[0]
    if len(data_vars) > 1:
        logger.debug("Multiple variables found (%s); using '%s'.", data_vars, var_name)

    da = ds_clip[var_name].load()

    # --- Strip NIMROD missing-data sentinels ---------------------------------
    # Existing on-disk files processed before the read_nimrod_file fix was
    # applied will not have a _FillValue attribute, so xarray cannot mask the
    # sentinel automatically.  Explicitly mask anything above the threshold.
    n_sentinel = int((da > NIMROD_FILL_THRESHOLD).sum())
    if n_sentinel > 0:
        logger.debug(
            "Masking %d sentinel value(s) (> %.0e) in loaded data.",
            n_sentinel,
            NIMROD_FILL_THRESHOLD,
        )
        da = da.where(da <= NIMROD_FILL_THRESHOLD)

    # --- Rename BNG projection coordinate names to short y / x ---------------
    rename_map = {k: v for k, v in {_Y: "y", _X: "x"}.items() if k in da.dims}
    if rename_map:
        da = da.rename(rename_map)

    # --- Build polygon mask --------------------------------------------------
    x_clipped = da.coords["x"].values
    y_clipped = da.coords["y"].values

    xx, yy = np.meshgrid(x_clipped, y_clipped)
    grid_points = shapely.points(xx.ravel(), yy.ravel())
    inside_mask = shapely.within(grid_points, polygon_bng).reshape(xx.shape)

    n_inside = int(inside_mask.sum())
    logger.info(
        "  Polygon covers %d / %d pixel(s) (%.1f %%) on the clipped grid.",
        n_inside,
        inside_mask.size,
        100.0 * n_inside / inside_mask.size if inside_mask.size > 0 else 0.0,
    )

    da_masked = da.where(inside_mask)

    # --- Finalise metadata ---------------------------------------------------
    da_masked.name = "rainfall_rate"
    da_masked.attrs = {
        "units": "mm/h",
        "long_name": "Rainfall rate",
        "crs": "EPSG:27700",
        "source": "NIMROD 1km composite",
    }

    n_times = da_masked.sizes.get("time", 0)
    logger.info(
        "Rainfall stored: %d timestep(s), grid %d (y) \u00d7 %d (x).",
        n_times,
        da_masked.sizes.get("y", 0),
        da_masked.sizes.get("x", 0),
    )

    return da_masked


# ---------------------------------------------------------------------------
# NIMROD 1km composite - constants
# ---------------------------------------------------------------------------

#: EPSG code for British National Grid, the native CRS of NIMROD 1km data.
NIMROD_BNG_EPSG: int = 27700

#: Base URL for the CEDA NIMROD 1km composite archive.
CEDA_NIMROD_BASE_URL: str = "https://dap.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km"

#: Filename pattern for the daily NIMROD tar files on CEDA.
NIMROD_FILE_PATTERN: str = "metoffice-c-band-rain-radar_uk_{date}_1km-composite.dat.gz.tar"


# ---------------------------------------------------------------------------
# NIMROD 1km composite - private CEDA download helpers
# ---------------------------------------------------------------------------


def _nimrod_generate_dates(start_date: date, end_date: date) -> list[date]:
    """Return every date from *start_date* to *end_date* inclusive."""
    dates: list[date] = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def _nimrod_create_session() -> requests.Session:
    """Return a ``requests.Session`` configured with CEDA credentials.

    Authentication is resolved in priority order:

    1. ``CEDA_TOKEN`` environment variable (Bearer token).
    2. ``~/.netrc`` entry for ``dap.ceda.ac.uk``.
    3. ``CEDA_USERNAME`` / ``CEDA_PASSWORD`` environment variables.
    """
    from netrc import NetrcParseError, netrc

    session = requests.Session()

    token = os.environ.get("CEDA_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
        logger.info("NIMROD: using CEDA Bearer token from $CEDA_TOKEN")
        return session

    try:
        auth_info = netrc().authenticators("dap.ceda.ac.uk")
        if auth_info:
            username, _, password = auth_info
            session.auth = (username, password)
            logger.info("NIMROD: using .netrc credentials for user: %s", username)
            return session
    except (FileNotFoundError, NetrcParseError, TypeError):
        pass

    username = os.environ.get("CEDA_USERNAME")
    password = os.environ.get("CEDA_PASSWORD")
    if username and password:
        session.auth = (username, password)
        logger.info("NIMROD: using env-var credentials for user: %s", username)
        return session

    logger.warning(
        "No CEDA credentials found.  Set CEDA_TOKEN, ~/.netrc (dap.ceda.ac.uk), "
        "or CEDA_USERNAME/CEDA_PASSWORD.  Downloads may fail."
    )
    return session


def _nimrod_build_url(year: int, date_obj: date) -> str:
    """Return the CEDA URL for the daily NIMROD tar file for *date_obj*."""
    date_str = date_obj.strftime("%Y%m%d")
    filename = NIMROD_FILE_PATTERN.format(date=date_str)
    return f"{CEDA_NIMROD_BASE_URL}/{year}/{filename}"


def _nimrod_download_tar(
    url: str,
    output_path: Path,
    session: requests.Session,
    timeout: int = 300,
    chunk_size: int = 8192,
) -> bool:
    """Download one NIMROD tar file from CEDA.

    Returns ``True`` on success, ``False`` on any failure.
    """
    try:
        response = session.get(url, stream=True, timeout=timeout, allow_redirects=True)

        if "auth.ceda.ac.uk" in response.url or "signin" in response.url.lower():
            logger.error("Authentication redirect for %s — check CEDA credentials.", url)
            return False

        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "html" in content_type:
            logger.error(
                "Received HTML from %s — likely an auth or access error (Content-Type: %s).",
                url,
                content_type,
            )
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        with open(output_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    downloaded += len(chunk)

        if downloaded == 0:
            logger.error("Empty download from %s", url)
            output_path.unlink(missing_ok=True)
            return False

        # Sanity-check: reject HTML masquerading as a tar file.
        with open(output_path, "rb") as fh:
            header = fh.read(512)
        if len(header) < 512 or header[:15].lower() == b"<!doctype html":
            logger.error("Downloaded file looks like HTML, not a tar: %s", url)
            output_path.unlink(missing_ok=True)
            return False

        logger.debug("Downloaded %s (%d bytes)", output_path.name, downloaded)
        return True

    except requests.exceptions.RequestException as exc:
        logger.error("Download failed for %s: %s", url, exc)
        output_path.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# NIMROD 1km composite - public processing utilities
# ---------------------------------------------------------------------------


def load_nimrod_shapefile(shapefile_path: Path) -> gpd.GeoDataFrame:
    """Load a shapefile and reproject to British National Grid (EPSG:27700).

    Parameters
    ----------
    shapefile_path :
        Path to a shapefile or a directory containing a single ``.shp`` file.
        Any input CRS is accepted — the data will be reprojected to BNG.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame in EPSG:27700 (British National Grid).
    """
    if shapefile_path.is_dir():
        shp_files = list(shapefile_path.glob("*.shp"))
        if not shp_files:
            raise ValueError(f"No .shp file found in {shapefile_path}")
        shapefile_path = shp_files[0]

    gdf = gpd.read_file(shapefile_path)

    if gdf.crs is None:
        logger.warning("Shapefile has no CRS defined, assuming BNG (EPSG:27700)")
        gdf = gdf.set_crs(epsg=NIMROD_BNG_EPSG)
    elif gdf.crs.to_epsg() != NIMROD_BNG_EPSG:
        logger.info("Reprojecting shapefile from %s to BNG", gdf.crs)
        gdf = gdf.to_crs(epsg=NIMROD_BNG_EPSG)

    return gdf


def extract_nimrod_tar(tar_path: Path, extract_dir: Path) -> list[Path]:
    """Extract a NIMROD ``.tar`` file and return paths to the ``.dat.gz`` members.

    CEDA archives are uncompressed tar files despite the ``.tar`` extension.
    Several open modes are tried in sequence so that compressed variants are
    also handled transparently.

    Parameters
    ----------
    tar_path :
        Path to the tar file.
    extract_dir :
        Directory to extract into (created if absent).

    Returns
    -------
    list[Path]
        Sorted list of extracted ``.dat.gz`` paths.
    """
    extracted: list[Path] = []
    try:
        opened = False
        for mode in ("r:", "r", "r:gz", "r:bz2", "r:xz"):
            try:
                with tarfile.open(tar_path, mode) as tf:  # type: ignore[arg-type]
                    opened = True
                    members = tf.getmembers()
                    logger.debug(
                        "Tar %s: %d member(s), mode=%s",
                        tar_path.name,
                        len(members),
                        mode,
                    )
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    tf.extractall(path=extract_dir)
                    for m in members:
                        if m.name.endswith(".dat.gz"):
                            extracted.append(extract_dir / m.name)
                    if not extracted:
                        names = [m.name for m in members[:20]]
                        logger.warning(
                            "No .dat.gz files in %s.  Members: %s",
                            tar_path.name,
                            names,
                        )
                    break
            except (tarfile.ReadError, tarfile.CompressionError) as exc:
                logger.debug("mode %s failed for %s: %s", mode, tar_path.name, exc)
        if not opened:
            logger.error("Could not open %s with any tar mode.", tar_path)
    except (tarfile.TarError, OSError) as exc:
        logger.error("Failed to extract %s: %s", tar_path, exc)
    return sorted(extracted)


#: NIMROD missing-data sentinel value.  The NIMROD binary format stores
#: missing/undetected cells as the integer code ``-32768``; after Iris applies
#: the per-file scale factor and offset this becomes a large float (~9.97e36).
#: Iris does not write this as ``_FillValue`` in the saved NetCDF, so it must
#: be masked explicitly.  Any value above this threshold is treated as missing.
NIMROD_FILL_THRESHOLD: float = 1e36


def read_nimrod_file(dat_gz_path: Path) -> iris.cube.Cube | None:
    """Decompress and load a NIMROD ``.dat.gz`` file as an Iris cube.

    Missing-data sentinel values (stored as large floats by Iris rather than
    as a NetCDF ``_FillValue`` attribute) are masked before the cube is
    returned, so that :func:`save_nimrod_cube` writes a proper
    ``_FillValue`` to disk.

    Parameters
    ----------
    dat_gz_path :
        Path to the compressed NIMROD binary file.

    Returns
    -------
    iris.cube.Cube or None
        Loaded cube with sentinel values masked, or ``None`` if reading fails.
    """
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as fh:
            tmp = Path(fh.name)
        with gzip.open(dat_gz_path, "rb") as gz, open(tmp, "wb") as out:
            shutil.copyfileobj(gz, out)
        cube = iris.load_cube(str(tmp))
        # Mask the NIMROD missing-data sentinel before returning so that
        # downstream saves write a proper _FillValue to NetCDF.
        cube.data = np.ma.masked_where(cube.data > NIMROD_FILL_THRESHOLD, cube.data)
        return cube
    except Exception as exc:
        logger.error("Failed to read %s: %s", dat_gz_path, exc)
        return None
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def _read_nimrod_gz_stream(
    gz_fileobj,
    name: str = "<stream>",
) -> iris.cube.Cube | None:
    """Load a NIMROD ``.dat.gz`` directly from a file-like object.

    Decompresses the gzip stream into a temporary ``.dat`` file (never
    touching the intermediate ``.dat.gz`` on disk) and loads it with Iris.
    The temporary file is removed before returning.

    Parameters
    ----------
    gz_fileobj :
        Readable file-like object containing gzip-compressed NIMROD data
        (e.g. from ``tarfile.TarFile.extractfile``).
    name :
        Descriptive label used only in error log messages.
    """
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as fh:
            tmp = Path(fh.name)
        with gzip.open(gz_fileobj, "rb") as gz, open(tmp, "wb") as out:
            shutil.copyfileobj(gz, out)  # type: ignore[misc]  # gz is GzipFile in "rb" mode
        return iris.load_cube(str(tmp))
    except Exception as exc:
        logger.error("Failed to read %s: %s", name, exc)
        return None
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def crop_nimrod_cube(
    cube: iris.cube.Cube,
    gdf: gpd.GeoDataFrame,
) -> iris.cube.Cube | None:
    """Crop an Iris cube to a shapefile extent and mask cells outside the polygons.

    The cube must use 1-D BNG projection coordinates (the native grid of
    NIMROD 1km composites).

    Parameters
    ----------
    cube :
        Iris cube with BNG projection coordinates.
    gdf :
        GeoDataFrame in BNG (EPSG:27700) whose union defines the mask.

    Returns
    -------
    iris.cube.Cube or None
        Spatially cropped and polygon-masked cube, or ``None`` on failure.
    """
    try:
        x_coord = cube.coord(axis="X")
        y_coord = cube.coord(axis="Y")

        if x_coord.ndim != 1 or y_coord.ndim != 1:
            logger.warning("crop_nimrod_cube: only 1-D coordinate grids are supported.")
            return None

        x_vals = x_coord.points
        y_vals = y_coord.points
        minx, miny, maxx, maxy = gdf.total_bounds

        x_idx = np.where((x_vals >= minx) & (x_vals <= maxx))[0]
        y_idx = np.where((y_vals >= miny) & (y_vals <= maxy))[0]

        if x_idx.size == 0 or y_idx.size == 0:
            logger.warning("Shapefile bounds do not intersect with cube extent.")
            return None

        x_slice = slice(x_idx[0], x_idx[-1] + 1)
        y_slice = slice(y_idx[0], y_idx[-1] + 1)

        x_dim = cube.coord_dims(x_coord)[0]
        y_dim = cube.coord_dims(y_coord)[0]

        if x_dim < y_dim:
            cropped = cube[:, x_slice, y_slice] if cube.ndim == 3 else cube[x_slice, y_slice]
        else:
            cropped = cube[:, y_slice, x_slice] if cube.ndim == 3 else cube[y_slice, x_slice]

        x_crop = cropped.coord(axis="X").points
        y_crop = cropped.coord(axis="Y").points
        x_res = float(np.median(np.diff(x_crop)))
        y_res = float(np.median(np.diff(y_crop)))

        transform = rasterio.transform.from_bounds(
            x_crop.min() - abs(x_res) / 2,
            y_crop.min() - abs(y_res) / 2,
            x_crop.max() + abs(x_res) / 2,
            y_crop.max() + abs(y_res) / 2,
            len(x_crop),
            len(y_crop),
        )

        mask = rasterio.features.geometry_mask(
            list(gdf.geometry),
            out_shape=(len(y_crop), len(x_crop)),
            transform=transform,
            invert=False,
        )

        # rasterio.transform.from_bounds is north-up (row 0 = north).
        # If the cube's y-axis is south-up (ascending values), flip the mask.
        if len(y_crop) > 1 and float(y_crop[0]) < float(y_crop[-1]):
            mask = mask[::-1, :]

        data = cropped.data
        if np.ma.is_masked(data):
            combined_mask = np.ma.getmaskarray(data) | mask
            data = np.ma.masked_array(np.ma.getdata(data), mask=combined_mask)
        else:
            data = np.ma.masked_array(data, mask=mask)
        cropped.data = data

        return cropped

    except Exception as exc:
        logger.error("crop_nimrod_cube failed: %s", exc)
        return None


def _crop_nimrod_cached(
    cube: iris.cube.Cube,
    gdf: gpd.GeoDataFrame,
    cache: _NimrodCropCache | None,
) -> tuple[iris.cube.Cube | None, _NimrodCropCache | None]:
    """Crop a NIMROD cube, building and caching spatial parameters on first call.

    All NIMROD timesteps share the same grid, so bounding-box slices and the
    rasterio geometry mask only need to be computed once per batch.  Pass
    ``cache=None`` on the first call; pass the returned cache object to
    every subsequent call in the same batch to skip recomputation.

    Parameters
    ----------
    cube :
        Iris cube with 1-D BNG projection coordinates.
    gdf :
        GeoDataFrame in BNG (EPSG:27700) defining the crop region.
    cache :
        ``None`` on the first call; the ``_NimrodCropCache`` returned by the
        previous call on all subsequent calls.

    Returns
    -------
    tuple
        ``(cropped_cube | None, cache | None)``
    """
    try:
        x_coord = cube.coord(axis="X")
        y_coord = cube.coord(axis="Y")

        if x_coord.ndim != 1 or y_coord.ndim != 1:
            logger.warning("_crop_nimrod_cached: only 1-D coordinate grids are supported.")
            return None, cache

        if cache is None:
            # First call: derive slices, compute and cache the geometry mask.
            x_vals = x_coord.points
            y_vals = y_coord.points
            minx, miny, maxx, maxy = gdf.total_bounds

            x_idx = np.where((x_vals >= minx) & (x_vals <= maxx))[0]
            y_idx = np.where((y_vals >= miny) & (y_vals <= maxy))[0]

            if x_idx.size == 0 or y_idx.size == 0:
                logger.warning("Shapefile bounds do not intersect with cube extent.")
                return None, None

            x_slice = slice(x_idx[0], x_idx[-1] + 1)
            y_slice = slice(y_idx[0], y_idx[-1] + 1)

            x_dim = cube.coord_dims(x_coord)[0]
            y_dim = cube.coord_dims(y_coord)[0]
            x_dim_first = x_dim < y_dim

            if x_dim_first:
                cropped = cube[:, x_slice, y_slice] if cube.ndim == 3 else cube[x_slice, y_slice]
            else:
                cropped = cube[:, y_slice, x_slice] if cube.ndim == 3 else cube[y_slice, x_slice]

            x_crop = cropped.coord(axis="X").points
            y_crop = cropped.coord(axis="Y").points
            x_res = float(np.median(np.diff(x_crop)))
            y_res = float(np.median(np.diff(y_crop)))

            transform = rasterio.transform.from_bounds(
                x_crop.min() - abs(x_res) / 2,
                y_crop.min() - abs(y_res) / 2,
                x_crop.max() + abs(x_res) / 2,
                y_crop.max() + abs(y_res) / 2,
                len(x_crop),
                len(y_crop),
            )
            mask = rasterio.features.geometry_mask(
                list(gdf.geometry),
                out_shape=(len(y_crop), len(x_crop)),
                transform=transform,
                invert=False,
            )
            if len(y_crop) > 1 and float(y_crop[0]) < float(y_crop[-1]):
                mask = mask[::-1, :]

            cache = _NimrodCropCache(
                x_slice=x_slice,
                y_slice=y_slice,
                ndim=cube.ndim,
                x_dim_first=x_dim_first,
                mask=mask,
            )
        else:
            # Subsequent calls: apply cached slices directly.
            if cache.x_dim_first:
                cropped = (
                    cube[:, cache.x_slice, cache.y_slice]
                    if cube.ndim == 3
                    else cube[cache.x_slice, cache.y_slice]
                )
            else:
                cropped = (
                    cube[:, cache.y_slice, cache.x_slice]
                    if cube.ndim == 3
                    else cube[cache.y_slice, cache.x_slice]
                )

        data = cropped.data
        if np.ma.is_masked(data):
            combined_mask = np.ma.getmaskarray(data) | cache.mask
            data = np.ma.masked_array(np.ma.getdata(data), mask=combined_mask)
        else:
            data = np.ma.masked_array(data, mask=cache.mask)
        cropped.data = data

        return cropped, cache

    except Exception as exc:
        logger.error("_crop_nimrod_cached failed: %s", exc)
        return None, cache


def save_nimrod_cube(cube: iris.cube.Cube, output_path: Path) -> bool:
    """Save an Iris cube to a NetCDF file.

    Parameters
    ----------
    cube :
        Iris cube to save.
    output_path :
        Destination ``.nc`` path.  Parent directories are created as needed.

    Returns
    -------
    bool
        ``True`` if the file was written successfully.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with _IRIS_WRITE_LOCK:
            # Re-assert inside the lock: iris FUTURE flags can misbehave
            # when accessed from worker threads for the first time.
            iris.FUTURE.save_split_attrs = True
            iris.FUTURE.date_microseconds = True
            iris.save(cube, str(output_path))
        return True
    except Exception as exc:
        logger.error("Failed to save %s: %s", output_path, exc)
        return False


# ---------------------------------------------------------------------------
# NIMROD 1km composite - file discovery (public)
# ---------------------------------------------------------------------------


def find_nimrod_nc_files(
    input_dir: Path,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[list[Path], int]:
    """Return sorted ``.nc`` paths under *input_dir*, optionally date-filtered.

    Files are expected to have stems of the form ``YYYYMMDD_HHMMSS``
    (per-timestep, written by :func:`process_nimrod_local`) or ``YYYYMMDD``
    (per-day, written by ``scripts/consolidate_nimrod.sh``) — a ``YYYYMMDD``
    stem is treated as covering the whole day when checking overlap with
    *start*/*end*, since the file itself may hold timesteps anywhere in that
    day. Any file whose stem matches neither format is always included.

    Parameters
    ----------
    input_dir :
        Root directory to search recursively.
    start :
        Optional inclusive lower bound.
    end :
        Optional inclusive upper bound.

    Returns
    -------
    tuple[list[Path], int]
        ``(matched_files, n_skipped)``
    """
    all_files = sorted(input_dir.rglob("*.nc"))
    if start is None and end is None:
        return all_files, 0

    matched: list[Path] = []
    skipped = 0
    for f in all_files:
        try:
            dt = datetime.strptime(f.stem, "%Y%m%d_%H%M%S")
            interval_end = dt
        except ValueError:
            try:
                dt = datetime.strptime(f.stem, "%Y%m%d")
                interval_end = dt + timedelta(days=1) - timedelta(microseconds=1)
            except ValueError:
                matched.append(f)
                continue
        if start is not None and interval_end < start:
            skipped += 1
            continue
        if end is not None and dt > end:
            skipped += 1
            continue
        matched.append(f)

    return matched, skipped


# ---------------------------------------------------------------------------
# NIMROD 1km composite - local tar processing (public)
# ---------------------------------------------------------------------------


def process_nimrod_tar(
    tar_path: Path,
    gdf: gpd.GeoDataFrame,
    output_dir: Path,
    temp_dir: Path,
) -> dict[str, int]:
    """Stream and process one locally downloaded NIMROD tar file.

    Opens the tar without full extraction, reads each ``.dat.gz`` member
    directly into memory, crops to *gdf* (reusing cached spatial parameters
    across timesteps), and writes per-timestep NetCDF files under
    ``output_dir/<year>/<YYYYMMDD>/``.

    Parameters
    ----------
    tar_path :
        Path to the ``.tar`` file.  The filename must follow the CEDA
        convention ``metoffice-c-band-rain-radar_uk_YYYYMMDD_...``.
    gdf :
        GeoDataFrame in BNG (EPSG:27700) used for spatial cropping.
    output_dir :
        Root output directory.
    temp_dir :
        Kept for API compatibility; no longer used for intermediate files.

    Returns
    -------
    dict
        ``{"processed": int, "errors": int}``
    """
    logger.info("Processing %s", tar_path.name)

    # Parse YYYYMMDD from e.g. metoffice-c-band-rain-radar_uk_20230101_...
    try:
        parts = tar_path.stem.split("_")  # stem strips .tar
        date_str = parts[2]  # YYYYMMDD
        year = date_str[:4]
    except (IndexError, ValueError) as exc:
        logger.error("Cannot parse date from %s: %s", tar_path.name, exc)
        return {"processed": 0, "errors": 1}

    date_output_dir = output_dir / year / date_str
    if date_output_dir.exists() and any(date_output_dir.glob("*.nc")):
        logger.info("  [skip] Already processed")
        return {"processed": 0, "errors": 0}

    processed = error_count = 0
    crop_cache: _NimrodCropCache | None = None
    opened = False

    for mode in ("r:", "r", "r:gz", "r:bz2", "r:xz"):
        try:
            with tarfile.open(tar_path, mode) as tf:  # type: ignore[arg-type]
                opened = True
                members = [m for m in tf.getmembers() if m.name.endswith(".dat.gz")]
                if not members:
                    names = [m.name for m in tf.getmembers()[:20]]
                    logger.warning("No .dat.gz files in %s.  Members: %s", tar_path.name, names)
                    return {"processed": 0, "errors": 1}

                for member in members:
                    gz_fileobj = tf.extractfile(member)
                    if gz_fileobj is None:
                        error_count += 1
                        continue

                    cube = _read_nimrod_gz_stream(gz_fileobj, member.name)
                    if cube is None:
                        error_count += 1
                        continue

                    cropped, crop_cache = _crop_nimrod_cached(cube, gdf, crop_cache)
                    if cropped is None:
                        error_count += 1
                        continue

                    try:
                        time_coord = cropped.coord("time")
                        ts = time_coord.units.num2date(time_coord.points[0])
                        timestamp_str = ts.strftime("%Y%m%d_%H%M%S")
                    except Exception:
                        timestamp_str = Path(member.name).stem.replace(".dat", "")

                    if save_nimrod_cube(cropped, date_output_dir / f"{timestamp_str}.nc"):
                        processed += 1
                    else:
                        error_count += 1
            break
        except (tarfile.ReadError, tarfile.CompressionError) as exc:
            logger.debug("mode %s failed for %s: %s", mode, tar_path.name, exc)

    if not opened:
        logger.error("Could not open %s with any tar mode.", tar_path)
        return {"processed": 0, "errors": 1}

    if error_count == 0:
        logger.info("  [done] Processed %d timestep(s)", processed)
    else:
        logger.warning("  [done] %d timestep(s) processed, %d error(s)", processed, error_count)

    return {"processed": processed, "errors": error_count}


def process_nimrod_local(
    input_paths: list[Path],
    shapefile_path: Path,
    output_dir: Path,
    max_workers: int | None = None,
) -> dict[str, int]:
    """Process locally downloaded NIMROD tar files and crop to a shapefile.

    Discovers ``.tar`` files in *input_paths* (files or directories), streams
    each one without full extraction, reads every ``.dat.gz`` timestep with
    Iris, crops it to *shapefile_path*, and writes per-timestep NetCDF files
    under ``output_dir/<year>/<YYYYMMDD>/``.  Multiple tar files are processed
    in parallel using a thread pool.

    Parameters
    ----------
    input_paths :
        List of ``.tar`` file paths or directories containing ``.tar`` files.
    shapefile_path :
        Path to the shapefile used for spatial cropping (any input CRS).
    output_dir :
        Root directory for processed NetCDF files.
    max_workers :
        Maximum number of parallel worker threads.  Defaults to
        ``min(len(tar_files), os.cpu_count())``.

    Returns
    -------
    dict
        ``{"processed": int, "errors": int}``
    """
    logger.info("NIMROD local file processor")
    logger.info("  Shapefile  : %s", shapefile_path)
    logger.info("  Output dir : %s", output_dir)

    gdf = load_nimrod_shapefile(shapefile_path)
    logger.info("  Bounds (BNG): %s", gdf.total_bounds)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)

    tar_files: list[Path] = []
    for p in input_paths:
        if p.is_file() and p.suffix == ".tar":
            tar_files.append(p)
        elif p.is_dir():
            tar_files.extend(p.glob("*.tar"))

    if not tar_files:
        logger.error("No tar files found in: %s", input_paths)
        return {"processed": 0, "errors": 0}

    tar_files = sorted(tar_files)
    workers = min(len(tar_files), max_workers or os.cpu_count() or 4)
    logger.info("Found %d tar file(s) to process (workers: %d)", len(tar_files), workers)
    results: dict[str, int] = {"processed": 0, "errors": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_nimrod_tar, tar_path, gdf, output_dir, temp_dir): tar_path
            for tar_path in tar_files
        }
        for future in as_completed(futures):
            tar_path = futures[future]
            try:
                r = future.result()
            except Exception as exc:
                logger.error(
                    "Unexpected error processing %s: %s", tar_path.name, exc, exc_info=True
                )
                results["errors"] += 1
                continue
            results["processed"] += r["processed"]
            results["errors"] += r["errors"]

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    logger.info("Done.  Processed: %d  Errors: %d", results["processed"], results["errors"])
    return results


# ---------------------------------------------------------------------------
# NIMROD 1km composite - CEDA download + process pipeline (public)
# ---------------------------------------------------------------------------


def _nimrod_process_date(
    date_obj: date,
    gdf: gpd.GeoDataFrame,
    output_dir: Path,
    temp_dir: Path,
    dry_run: bool,
    print_lock: threading.Lock,
    session: requests.Session,
) -> dict[str, int]:
    """Download and process one day of NIMROD data (worker function)."""
    date_str = date_obj.strftime("%Y%m%d")
    year = date_obj.year
    url = _nimrod_build_url(year, date_obj)

    with print_lock:
        logger.info("Processing %s", date_str)

    if dry_run:
        with print_lock:
            logger.info("  [dry-run] %s", url)
        return {"processed": 0, "skipped": 0, "errors": 0, "dry_run": 1}

    date_output_dir = output_dir / str(year) / date_str
    if date_output_dir.exists() and any(date_output_dir.glob("*.nc")):
        with print_lock:
            logger.info("  [skip] Already processed")
        return {"processed": 0, "skipped": 1, "errors": 0, "dry_run": 0}

    tar_path = temp_dir / f"{date_str}.tar"
    if not _nimrod_download_tar(url, tar_path, session):
        with print_lock:
            logger.error("  [error] Download failed: %s", url)
        return {"processed": 0, "skipped": 0, "errors": 1, "dry_run": 0}

    processed = error_count = 0
    crop_cache: _NimrodCropCache | None = None
    opened = False

    for mode in ("r:", "r", "r:gz", "r:bz2", "r:xz"):
        try:
            with tarfile.open(tar_path, mode) as tf:  # type: ignore[arg-type]
                opened = True
                members = [m for m in tf.getmembers() if m.name.endswith(".dat.gz")]
                if not members:
                    with print_lock:
                        logger.error("  [error] No .dat.gz files in tar for %s", date_str)
                    tar_path.unlink(missing_ok=True)
                    return {"processed": 0, "skipped": 0, "errors": 1, "dry_run": 0}

                for member in members:
                    gz_fileobj = tf.extractfile(member)
                    if gz_fileobj is None:
                        error_count += 1
                        continue

                    cube = _read_nimrod_gz_stream(gz_fileobj, member.name)
                    if cube is None:
                        error_count += 1
                        continue

                    cropped, crop_cache = _crop_nimrod_cached(cube, gdf, crop_cache)
                    if cropped is None:
                        error_count += 1
                        continue

                    try:
                        time_coord = cropped.coord("time")
                        ts = time_coord.units.num2date(time_coord.points[0])
                        timestamp_str = ts.strftime("%Y%m%d_%H%M%S")
                    except Exception:
                        timestamp_str = Path(member.name).stem.replace(".dat", "")

                    if save_nimrod_cube(cropped, date_output_dir / f"{timestamp_str}.nc"):
                        processed += 1
                    else:
                        error_count += 1
            break
        except (tarfile.ReadError, tarfile.CompressionError):
            pass

    tar_path.unlink(missing_ok=True)

    if not opened:
        with print_lock:
            logger.error("  [error] Could not open tar for %s", date_str)
        return {"processed": 0, "skipped": 0, "errors": 1, "dry_run": 0}

    with print_lock:
        if error_count == 0:
            logger.info("  [done] %d timestep(s)", processed)
        else:
            logger.warning("  [done] %d timestep(s), %d error(s)", processed, error_count)

    return {"processed": processed, "skipped": 0, "errors": error_count, "dry_run": 0}


def download_and_crop_nimrod(
    years: list[int],
    shapefile_path: Path,
    output_dir: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    workers: int = 4,
    dry_run: bool = False,
) -> dict[str, int]:
    """Download NIMROD 1km composite data from CEDA and crop to a shapefile.

    Fetches the daily tar files from the CEDA NIMROD archive for each
    requested year, extracts every ``.dat.gz`` timestep with Iris, crops it
    to *shapefile_path*, and writes per-timestep NetCDF files under
    ``output_dir/<year>/<YYYYMMDD>/``.

    Parameters
    ----------
    years :
        Calendar years to download (e.g. ``[2023, 2024]``).
    shapefile_path :
        Path to the shapefile used for spatial cropping (any input CRS).
    output_dir :
        Root directory for the processed NetCDF files.
    start_date :
        Optional inclusive start date filter.
    end_date :
        Optional inclusive end date filter.
    workers :
        Number of parallel download/processing threads.
    dry_run :
        If ``True``, log what would be downloaded without doing anything.

    Returns
    -------
    dict
        ``{"processed": int, "skipped": int, "errors": int, "dry_run": int}``
    """
    logger.info("NIMROD 1km composite downloader and cropper")
    logger.info("  Years      : %s", years)
    logger.info("  Shapefile  : %s", shapefile_path)
    logger.info("  Output dir : %s", output_dir)
    logger.info("  Workers    : %d", workers)
    logger.info("  Dry run    : %s", dry_run)

    gdf = load_nimrod_shapefile(shapefile_path)
    logger.info("  Bounds (BNG): %s", gdf.total_bounds)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)

    dates_to_process: list[date] = []
    for year in years:
        y_start = start_date if (start_date and start_date.year == year) else date(year, 1, 1)
        y_end = end_date if (end_date and end_date.year == year) else date(year, 12, 31)
        dates_to_process.extend(_nimrod_generate_dates(y_start, y_end))

    logger.info(
        "Processing %d date(s) across %d year(s)",
        len(dates_to_process),
        len(years),
    )

    session = _nimrod_create_session()
    print_lock = threading.Lock()
    results: dict[str, int] = {"processed": 0, "skipped": 0, "errors": 0, "dry_run": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _nimrod_process_date,
                d,
                gdf,
                output_dir,
                temp_dir,
                dry_run,
                print_lock,
                session,
            ): d
            for d in dates_to_process
        }
        for future in as_completed(futures):
            r = future.result()
            for k, v in r.items():
                results[k] = results.get(k, 0) + v

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    logger.info("Done.")
    logger.info("  Processed : %d", results["processed"])
    logger.info("  Skipped   : %d", results["skipped"])
    logger.info("  Errors    : %d", results["errors"])
    if dry_run:
        logger.info("  Dry-run   : %d", results["dry_run"])

    return results
