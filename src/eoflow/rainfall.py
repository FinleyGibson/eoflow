"""
rainfall.py

Met Office UKV 2 km rainfall-rate data access for the eoflow package.

This module provides two complementary capabilities:

1. **Downloading** – fetch ``rainfall_rate.nc`` NetCDF files from the
   AWS Open Data S3 bucket (no credentials required).
2. **Converting** – reproject those NetCDF files from the native Lambert
   Azimuthal Equal Area grid onto a British National Grid (EPSG:27700)
   regular grid and write ESRI ASCII raster (``.asc``) files.

Both capabilities are exposed as importable Python functions suitable for
use in notebooks and pipelines, as well as thin CLI wrappers in the
``scripts/`` directory.

Bucket
------
``s3://met-office-atmospheric-model-data/uk-deterministic-2km/``

No AWS account or credentials are required (``--no-sign-request``).

File-naming convention on S3
-----------------------------
``uk-deterministic-2km/<run_time>/<valid_time>-PT<lead>H<mm>M-rainfall_rate.nc``

For example::

    20240301T0000Z/20240301T0000Z-PT0000H00M-rainfall_rate.nc
    20240301T0000Z/20240301T0015Z-PT0000H15M-rainfall_rate.nc
    20240301T0000Z/20240301T0100Z-PT0001H00M-rainfall_rate.nc

Output layout from :func:`convert_rainfall`
--------------------------------------------
``<output_dir>/<run_YYYYMMDDHHMM>/<valid_YYYYMMDDHHMM>.asc``

This matches the naming convention used by ``trigger-aws.py`` so the
``.asc`` files can be dropped straight into the same downstream workflow.
"""

from __future__ import annotations

import gzip
import shutil
import tarfile
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import boto3
import geopandas as gpd
import iris
import iris.analysis
import iris.coord_systems
import iris.coords
import iris.cube
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import requests
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from rasterio.transform import from_origin

from eoflow.log_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: S3 bucket name hosting the Met Office Open Data.
BUCKET = "met-office-atmospheric-model-data"

#: Key prefix for the UKV 2 km deterministic model data.
PREFIX = "uk-deterministic-2km"

#: Suffix shared by all rainfall-rate NetCDF files.
NC_SUFFIX = "-rainfall_rate.nc"

#: Temporal resolution of valid-time steps available on S3 (minutes).
TIMESTEP_MINUTES = 15

#: Maximum lead time of any model run that is available on S3 (hours).
MAX_LEAD_HOURS = 54

#: Default output raster resolution in metres.
DEFAULT_RESOLUTION_M = 1000

#: Default BNG extent (metres) covering the full UK.
UK_BNG_EXTENT = (0.0, 0.0, 700_000.0, 1_300_000.0)

# Model-run hours present in the S3 bucket.
# Nowcast runs: 01,02,04,05,07,08,10,11,13,14,16,17,19,20,22,23
# Short runs  : 00,06,09,12,18,21
# Medium runs : 03,15
ALL_RUN_HOURS: list[int] = sorted(range(24))


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def make_s3_client():
    """Return an anonymous (unsigned) boto3 S3 client for ``eu-west-2``.

    No AWS account or credentials are required to access the Met Office
    Open Data bucket.

    Returns
    -------
    botocore.client.S3
        Configured S3 client.
    """
    return boto3.client(
        "s3",
        region_name="eu-west-2",
        config=Config(signature_version=UNSIGNED),
    )


def s3_key(run_dt: datetime, valid_dt: datetime) -> str:
    """Construct the S3 object key for a given model-run time and valid time.

    Parameters
    ----------
    run_dt :
        Model-run datetime (UTC, minute precision).
    valid_dt :
        Valid (forecast target) datetime (UTC, minute precision).

    Returns
    -------
    str
        Full S3 key, e.g.
        ``uk-deterministic-2km/20240301T0000Z/20240301T0015Z-PT0000H15M-rainfall_rate.nc``.
    """
    lead = valid_dt - run_dt
    total_seconds = int(lead.total_seconds())
    lead_hours, remainder = divmod(total_seconds, 3600)
    lead_minutes = remainder // 60

    run_str = run_dt.strftime("%Y%m%dT%H%MZ")
    valid_str = valid_dt.strftime("%Y%m%dT%H%MZ")
    lead_str = f"PT{lead_hours:04d}H{lead_minutes:02d}M"

    filename = f"{valid_str}-{lead_str}{NC_SUFFIX}"
    return f"{PREFIX}/{run_str}/{filename}"


def _object_exists(s3, key: str) -> bool:
    """Return ``True`` if *key* exists in :data:`BUCKET` (head-object only)."""
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Download: core logic
# ---------------------------------------------------------------------------


def candidate_run_times(
    valid_dt: datetime,
    run_hour_filter: int | None = None,
):
    """Yield candidate model-run datetimes that could contain *valid_dt*.

    Results are ordered newest-run-first (prefer the most recent run for a
    given valid time step).

    A run at time R can forecast valid time V only when ``R <= V`` and
    ``(V - R) <= MAX_LEAD_HOURS`` hours.

    Parameters
    ----------
    valid_dt :
        The target valid time (UTC, timezone-aware).
    run_hour_filter :
        If provided, only yield run datetimes whose hour matches this value.

    Yields
    ------
    datetime
        Candidate model-run datetimes in descending order.
    """
    seen: set[datetime] = set()
    for hours_back in range(MAX_LEAD_HOURS + 1):
        candidate = valid_dt - timedelta(hours=hours_back)
        candidate = candidate.replace(minute=0, second=0, microsecond=0)
        if candidate.hour not in ALL_RUN_HOURS:
            continue
        if run_hour_filter is not None and candidate.hour != run_hour_filter:
            continue
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def find_s3_key_for_valid_time(
    s3,
    valid_dt: datetime,
    run_hour_filter: int | None = None,
) -> str | None:
    """Try candidate run times for *valid_dt* and return the first existing S3 key.

    Parameters
    ----------
    s3 :
        Boto3 S3 client (anonymous is fine).
    valid_dt :
        Target valid time (UTC, timezone-aware).
    run_hour_filter :
        Optional model-run hour filter (0–23).

    Returns
    -------
    str or None
        The S3 key if a matching file is found, otherwise ``None``.
    """
    for run_dt in candidate_run_times(valid_dt, run_hour_filter):
        key = s3_key(run_dt, valid_dt)
        if _object_exists(s3, key):
            return key
    return None


def _download_key(
    s3,
    key: str,
    output_dir: Path,
    dry_run: bool,
    lock: threading.Lock,
) -> tuple[str, str]:
    """Download a single S3 object, preserving its sub-directory structure.

    Parameters
    ----------
    s3 :
        Boto3 S3 client.
    key :
        S3 object key to download.
    output_dir :
        Root directory under which the key's relative path is mirrored.
    dry_run :
        If ``True``, log what would be done without downloading.
    lock :
        Threading lock used to serialise console output.

    Returns
    -------
    tuple[str, str]
        ``(key, status)`` where *status* is one of ``'downloaded'``,
        ``'skipped'``, ``'dry-run'``, or ``'error:<msg>'``.
    """
    local_path = output_dir / key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        with lock:
            logger.info("  [dry-run] would download: s3://%s/%s", BUCKET, key)
        return key, "dry-run"

    if local_path.exists():
        with lock:
            logger.info("  [skip]   already exists: %s", local_path)
        return key, "skipped"

    try:
        s3.download_file(BUCKET, key, str(local_path))
        with lock:
            logger.info("  [ok]     %s", local_path)
        return key, "downloaded"
    except Exception as exc:
        with lock:
            logger.error("  [error]  %s: %s", key, exc)
        return key, f"error:{exc}"


def generate_valid_times(start: datetime, end: datetime):
    """Yield every 15-minute UTC step between *start* and *end* (inclusive).

    *start* is rounded up to the nearest :data:`TIMESTEP_MINUTES` boundary
    if necessary.

    Parameters
    ----------
    start, end :
        UTC datetimes (timezone-aware).

    Yields
    ------
    datetime
        Valid times at 15-minute intervals.
    """
    remainder = start.minute % TIMESTEP_MINUTES
    if remainder:
        start = start + timedelta(minutes=(TIMESTEP_MINUTES - remainder))
    start = start.replace(second=0, microsecond=0)

    current = start
    while current <= end:
        yield current
        current += timedelta(minutes=TIMESTEP_MINUTES)


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
# Download: public API
# ---------------------------------------------------------------------------


def download_rainfall(
    start: str | datetime,
    end: str | datetime,
    output_dir: Path | str = "./rainfall_data",
    *,
    run_hour: int | None = None,
    workers: int = 4,
    dry_run: bool = False,
) -> dict[str, int]:
    """Download Met Office UKV ``rainfall_rate`` NetCDF files from AWS S3.

    Files are saved under *output_dir*, mirroring the S3 key structure::

        <output_dir>/uk-deterministic-2km/<run_time>/<valid_time>-...-rainfall_rate.nc

    No AWS credentials are required.

    Parameters
    ----------
    start :
        Start of the valid-time range (UTC, inclusive).  Either a
        ``YYYY-MM-DDTHH:MM`` string or an aware :class:`~datetime.datetime`.
    end :
        End of the valid-time range (UTC, inclusive).  Same format as *start*.
    output_dir :
        Local directory under which files are saved.  Created if absent.
    run_hour :
        If set, only use files from the model run starting at this UTC hour
        (0–23).  Useful to pin to a specific run type.
    workers :
        Number of parallel download threads.
    dry_run :
        If ``True``, log what would be downloaded without downloading anything.

    Returns
    -------
    dict[str, int]
        Counts keyed by ``'downloaded'``, ``'skipped'``, ``'dry-run'``,
        and ``'error'``.

    Raises
    ------
    ValueError
        If *end* is before *start*.

    Examples
    --------
    >>> from pathlib import Path
    >>> from eoflow.rainfall import download_rainfall
    >>> counts = download_rainfall(
    ...     start="2024-03-01T00:00",
    ...     end="2024-03-01T06:00",
    ...     output_dir=Path("data/rainfall"),
    ... )
    >>> print(counts)
    """
    if isinstance(start, str):
        start = parse_datetime(start)
    if isinstance(end, str):
        end = parse_datetime(end)

    if end < start:
        raise ValueError("'end' must be >= 'start'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Met Office UKV rainfall downloader")
    logger.info("  Start   : %s", start.isoformat())
    logger.info("  End     : %s", end.isoformat())
    logger.info("  Output  : %s", output_dir.resolve())
    logger.info("  Workers : %d", workers)
    logger.info("  Dry run : %s", dry_run)
    if run_hour is not None:
        logger.info("  Run hour filter: %02dZ", run_hour)

    s3 = make_s3_client()
    print_lock = threading.Lock()

    # Phase 1: resolve valid times -> S3 keys
    valid_times = list(generate_valid_times(start, end))
    logger.info("Resolving %d valid time steps against S3 ...", len(valid_times))

    keys_to_download: list[str] = []
    missing: list[datetime] = []

    def _resolve(vt: datetime) -> tuple[datetime, str | None]:
        key = find_s3_key_for_valid_time(s3, vt, run_hour)
        return vt, key

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_resolve, vt): vt for vt in valid_times}
        for future in as_completed(futures):
            vt, key = future.result()
            if key:
                keys_to_download.append(key)
            else:
                missing.append(vt)
                logger.warning("  [warn] no S3 file found for valid time %s", vt.isoformat())

    keys_to_download.sort()
    logger.info(
        "Found %d files to download (%d valid times had no matching file).",
        len(keys_to_download),
        len(missing),
    )

    if not keys_to_download:
        logger.info("Nothing to download.")
        return {"downloaded": 0, "skipped": 0, "dry-run": 0, "error": 0}

    # Phase 2: download
    results: dict[str, int] = {"downloaded": 0, "skipped": 0, "dry-run": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_key, s3, key, output_dir, dry_run, print_lock): key
            for key in keys_to_download
        }
        for future in as_completed(futures):
            _, status = future.result()
            kind = "error" if status.startswith("error") else status
            results[kind] = results.get(kind, 0) + 1

    logger.info("Done.")
    logger.info("  Downloaded : %d", results["downloaded"])
    logger.info("  Skipped    : %d", results["skipped"])
    logger.info("  Dry-run    : %d", results["dry-run"])
    logger.info("  Errors     : %d", results["error"])

    return results


# ---------------------------------------------------------------------------
# Convert: helpers
# ---------------------------------------------------------------------------


def find_netcdf_files(input_dir: Path) -> list[Path]:
    """Recursively find all ``*rainfall_rate.nc`` files under *input_dir*.

    Parameters
    ----------
    input_dir :
        Root directory to search.

    Returns
    -------
    list[Path]
        Sorted list of matching paths.
    """
    return sorted(input_dir.rglob(f"*{NC_SUFFIX}"))


def parse_timestamps_from_path(nc_path: Path) -> tuple[str | None, str | None]:
    """Extract ``(run_timestamp_str, valid_timestamp_str)`` from a NetCDF path.

    Expected filename format::

        <valid_YYYYMMDDTHHMMZ>-PT<HHHHH>H<MM>M-rainfall_rate.nc

    The run timestamp is the name of the *parent directory*::

        <run_YYYYMMDDTHHMMZ>/

    Returns
    -------
    tuple[str or None, str or None]
        Both timestamps as ``'YYYYMMDDHHMM'`` strings suitable for use in
        output directory / file names.  Returns ``(None, None)`` if parsing
        fails.
    """
    try:
        run_dir = nc_path.parent.name  # e.g. 20240301T0000Z
        valid_part = nc_path.name.split("-")[0]  # e.g. 20240301T0015Z

        def _compact(s: str) -> str:
            return s.replace("T", "").replace("Z", "")

        return _compact(run_dir), _compact(valid_part)
    except Exception:
        return None, None


def output_path_for(nc_path: Path, output_dir: Path) -> Path:
    """Derive the ``.asc`` output path for a given input NetCDF file.

    Layout: ``<output_dir>/<run_YYYYMMDDHHMM>/<valid_YYYYMMDDHHMM>.asc``

    Falls back to a flat layout if the timestamps cannot be parsed.

    Parameters
    ----------
    nc_path :
        Source NetCDF file path.
    output_dir :
        Root output directory.

    Returns
    -------
    Path
        Destination ``.asc`` path.
    """
    run_str, valid_str = parse_timestamps_from_path(nc_path)
    if run_str is None:
        return output_dir / nc_path.with_suffix(".asc").name
    return output_dir / run_str / f"{valid_str}.asc"


def resolve_extent(
    extent: tuple[float, float, float, float] | None,
    input_files: list[Path],
) -> tuple[float, float, float, float]:
    """Return ``(x_min, y_min, x_max, y_max)`` in BNG metres.

    If *extent* is provided it is returned as-is.  Otherwise the UK-wide
    BNG defaults (:data:`UK_BNG_EXTENT`) are used with a warning.

    Parameters
    ----------
    extent :
        An explicit ``(x_min, y_min, x_max, y_max)`` tuple, or ``None``.
    input_files :
        List of discovered NetCDF files (used only in the warning message).

    Returns
    -------
    tuple[float, float, float, float]
        ``(x_min, y_min, x_max, y_max)`` in BNG metres.
    """
    if extent is not None:
        return tuple(float(v) for v in extent)  # type: ignore[return-value]

    if not input_files:
        logger.warning("No input files found and no extent given; using UK-wide BNG defaults.")
    else:
        logger.info(
            "No extent supplied; using UK-wide BNG defaults %s. "
            "Pass an explicit extent to restrict the domain.",
            UK_BNG_EXTENT,
        )
    return UK_BNG_EXTENT


# ---------------------------------------------------------------------------
# Convert: core conversion
# ---------------------------------------------------------------------------


def convert_netcdf_to_asc(
    input_path: Path,
    output_path: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    res: int = DEFAULT_RESOLUTION_M,
) -> None:
    """Reproject a UKV ``rainfall_rate`` NetCDF file to BNG and write an ASC raster.

    The rainfall rate is converted from m/s to mm/h (×3 600 000) before
    writing.

    Parameters
    ----------
    input_path :
        Source NetCDF file.
    output_path :
        Destination ``.asc`` file.  Parent directories are created as needed.
    x_min, x_max :
        Easting extent in BNG metres (EPSG:27700).
    y_min, y_max :
        Northing extent in BNG metres (EPSG:27700).
    res :
        Output pixel size in metres (default: 1000).
    """
    # Suppress iris split-attributes deprecation warning
    iris.FUTURE.save_split_attrs = True

    # Load and convert units (m/s -> mm/h)
    cube = iris.load_cube(str(input_path))
    cube.data = cube.data * 3_600_000

    # Build BNG target grid
    eastings = np.arange(x_min, x_max, res)
    northings = np.arange(y_min, y_max, res)

    bng_cs = iris.coord_systems.OSGB()
    x_coord = iris.coords.DimCoord(
        eastings,
        standard_name="projection_x_coordinate",
        units="m",
        coord_system=bng_cs,
    )
    y_coord = iris.coords.DimCoord(
        northings,
        standard_name="projection_y_coordinate",
        units="m",
        coord_system=bng_cs,
    )
    target_cube = iris.cube.Cube(
        np.zeros((len(northings), len(eastings)), np.float32),
        dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
    )

    # Regrid onto BNG
    bng_cube = cube.regrid(target_cube, iris.analysis.Linear())

    # ASCII rasters are top-down; NetCDF data is bottom-up, so flip Y.
    data_array = bng_cube.data[::-1, :]

    # from_origin takes the top-left corner (west edge, north edge)
    transform = from_origin(eastings[0], northings[-1], res, res)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(output_path),
        "w",
        driver="AAIGrid",
        height=data_array.shape[0],
        width=data_array.shape[1],
        count=1,
        dtype=data_array.dtype,
        crs="EPSG:27700",
        transform=transform,
    ) as dst:
        dst.write(data_array, 1)


def _convert_one(
    nc_path: Path,
    output_dir: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    res: int,
    dry_run: bool,
) -> tuple[Path, str]:
    """Convert a single NetCDF file (runs in a worker process).

    Returns
    -------
    tuple[Path, str]
        ``(nc_path, status)`` where *status* is one of ``'converted:<asc_path>'``,
        ``'skipped:<asc_path>'``, ``'dry-run:<asc_path>'``, or
        ``'error:<msg>'``.
    """
    asc_path = output_path_for(nc_path, output_dir)

    if dry_run:
        return nc_path, f"dry-run:{asc_path}"

    if asc_path.exists():
        return nc_path, f"skipped:{asc_path}"

    try:
        convert_netcdf_to_asc(nc_path, asc_path, x_min, x_max, y_min, y_max, res)
        return nc_path, f"converted:{asc_path}"
    except Exception as exc:
        return nc_path, f"error:{exc}"


# ---------------------------------------------------------------------------
# Convert: public API
# ---------------------------------------------------------------------------


def convert_rainfall(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    extent: tuple[float, float, float, float] | None = None,
    res: int = DEFAULT_RESOLUTION_M,
    workers: int = 4,
    dry_run: bool = False,
) -> dict[str, int]:
    """Convert UKV rainfall-rate NetCDF files to BNG ASCII rasters.

    Searches *input_dir* recursively for files matching ``*rainfall_rate.nc``,
    reprojects each one to British National Grid (EPSG:27700) at the requested
    resolution, and writes ESRI ASCII raster (``.asc``) files under
    *output_dir*.

    The output directory layout mirrors the naming convention used by
    ``trigger-aws.py``::

        <output_dir>/<run_YYYYMMDDHHMM>/<valid_YYYYMMDDHHMM>.asc

    Parameters
    ----------
    input_dir :
        Root directory containing downloaded ``*rainfall_rate.nc`` files
        (searched recursively).
    output_dir :
        Directory to write ``.asc`` files into.
    extent :
        Spatial extent as ``(x_min, y_min, x_max, y_max)`` in BNG metres
        (EPSG:27700).  Defaults to the full-UK extent
        ``(0, 0, 700000, 1300000)``.
    res :
        Output grid resolution in metres (default: 1000).
    workers :
        Number of parallel conversion processes.
    dry_run :
        If ``True``, log what would be converted without doing it.

    Returns
    -------
    dict[str, int]
        Counts keyed by ``'converted'``, ``'skipped'``, ``'dry-run'``,
        and ``'error'``.

    Raises
    ------
    FileNotFoundError
        If *input_dir* does not exist.

    Examples
    --------
    >>> from pathlib import Path
    >>> from eoflow.rainfall import convert_rainfall
    >>> counts = convert_rainfall(
    ...     input_dir=Path("data/rainfall"),
    ...     output_dir=Path("data/rainfall_asc"),
    ...     extent=(285000, 54000, 295000, 70000),
    ...     res=1000,
    ... )
    >>> print(counts)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Met Office UKV rainfall converter")
    logger.info("  Input   : %s", input_dir.resolve())
    logger.info("  Output  : %s", output_dir.resolve())
    logger.info("  Res     : %d m", res)
    logger.info("  Workers : %d", workers)
    logger.info("  Dry run : %s", dry_run)

    # Discover files
    logger.info("Scanning for NetCDF files ...")
    nc_files = find_netcdf_files(input_dir)
    logger.info("  Found %d file(s).", len(nc_files))

    if not nc_files:
        logger.info("Nothing to convert.")
        return {"converted": 0, "skipped": 0, "dry-run": 0, "error": 0}

    # Resolve spatial extent
    x_min, y_min, x_max, y_max = resolve_extent(extent, nc_files)
    logger.info(
        "Extent (BNG): x=[%g, %g]  y=[%g, %g]  res=%d m",
        x_min,
        x_max,
        y_min,
        y_max,
        res,
    )

    # Convert in parallel
    results: dict[str, int] = {"converted": 0, "skipped": 0, "dry-run": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _convert_one,
                nc_path,
                output_dir,
                x_min,
                x_max,
                y_min,
                y_max,
                res,
                dry_run,
            ): nc_path
            for nc_path in nc_files
        }
        for future in as_completed(futures):
            nc_path, status = future.result()
            kind, detail = (status.split(":", 1) + [""])[:2]
            if kind == "converted":
                results["converted"] += 1
                logger.info("  [ok]      %s", detail)
            elif kind == "skipped":
                results["skipped"] += 1
                logger.info("  [skip]    %s", detail)
            elif kind == "dry-run":
                results["dry-run"] += 1
                logger.info("  [dry-run] %s", detail)
            else:
                results["error"] += 1
                logger.error("  [error]   %s: %s", futures[future], detail)

    logger.info("Done.")
    logger.info("  Converted : %d", results["converted"])
    logger.info("  Skipped   : %d", results["skipped"])
    logger.info("  Dry-run   : %d", results["dry-run"])
    logger.info("  Errors    : %d", results["error"])

    return results


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


def _load_cube_for_polygon(
    nc_path: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    res: int,
) -> np.ndarray:
    """Load a single rainfall NetCDF and reproject onto a BNG target grid.

    The rainfall rate is converted from m/s to mm/h (× 3 600 000).  Any
    fill/mask values introduced by the regridding step are replaced with
    ``NaN``.

    Parameters
    ----------
    nc_path :
        Source NetCDF file.
    x_min, x_max :
        Easting extent in BNG metres (EPSG:27700).
    y_min, y_max :
        Northing extent in BNG metres (EPSG:27700).
    res :
        Output pixel size in metres.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(len(northings), len(eastings))`` in mm/h,
        with ``NaN`` where the regridder could not interpolate.
    """
    iris.FUTURE.save_split_attrs = True

    cube = iris.load_cube(str(nc_path))
    cube.data = cube.data * 3_600_000  # m/s -> mm/h

    eastings = np.arange(x_min, x_max, res, dtype=np.float64)
    northings = np.arange(y_min, y_max, res, dtype=np.float64)

    bng_cs = iris.coord_systems.OSGB()
    x_coord = iris.coords.DimCoord(
        eastings,
        standard_name="projection_x_coordinate",
        units="m",
        coord_system=bng_cs,
    )
    y_coord = iris.coords.DimCoord(
        northings,
        standard_name="projection_y_coordinate",
        units="m",
        coord_system=bng_cs,
    )
    target_cube = iris.cube.Cube(
        np.zeros((len(northings), len(eastings)), np.float32),
        dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
    )

    bng_cube = cube.regrid(target_cube, iris.analysis.Linear())

    # Collapse any masked array to a plain float32 ndarray, filling with NaN.
    raw = bng_cube.data
    if hasattr(raw, "filled"):
        data = raw.filled(np.nan).astype(np.float32)
    else:
        data = np.asarray(raw, dtype=np.float32)

    return data


def get_rainfall_for_polygon(
    polygon,
    start: str | datetime,
    end: str | datetime,
    *,
    download_dir: Path | str | None = None,
    res: int = DEFAULT_RESOLUTION_M,
    run_hour: int | None = None,
    workers: int = 4,
) -> xr.DataArray:
    """Get rainfall rate data clipped to a polygon for a given time range.

    Downloads UKV 2 km rainfall-rate NetCDF files from the Met Office AWS S3
    bucket for the requested period, reprojects each file from the native
    Lambert Azimuthal Equal Area grid onto a British National Grid
    (EPSG:27700) regular grid bounded by the polygon's extent, then masks
    all grid cells whose centre lies **outside** the polygon to ``NaN``.

    Parameters
    ----------
    polygon :
        Area of interest in **WGS 84 (EPSG:4326)**.  May be a
        ``shapely.geometry.Polygon`` or ``MultiPolygon`` — for example, a
        catchment boundary returned by
        :func:`eoflow.catchment.delineate_catchment`.
    start :
        Start of the valid-time range (UTC, inclusive).  Either a
        ``YYYY-MM-DDTHH:MM`` string or an aware :class:`~datetime.datetime`.
    end :
        End of the valid-time range (UTC, inclusive).  Same format as *start*.
    download_dir :
        Directory in which to store downloaded ``*.nc`` files.  If ``None``
        a temporary directory is created and removed automatically once the
        data has been loaded into memory.  Pass an explicit directory to
        cache downloads across repeated calls.
    res :
        Output grid resolution in BNG metres
        (default: :data:`DEFAULT_RESOLUTION_M`).
    run_hour :
        If set, restrict to files from the model run starting at this UTC
        hour (0–23).  See :func:`download_rainfall` for details.
    workers :
        Number of parallel download threads.

    Returns
    -------
    xarray.DataArray
        Rainfall rate in **mm/h** with dimensions ``(time, y, x)``:

        - ``time``  – UTC :class:`~datetime.datetime` of each valid timestep.
        - ``y``     – BNG northings (metres, EPSG:27700), south-to-north.
        - ``x``     – BNG eastings (metres, EPSG:27700), west-to-east.

        Grid cells whose centre lies **outside** *polygon* are ``NaN``.

        ``attrs`` on the returned array:

        - ``units``      ``"mm/h"``
        - ``long_name``  ``"Rainfall rate"``
        - ``crs``        ``"EPSG:27700"``
        - ``source``     ``"Met Office UKV 2km deterministic model"``

        Returns an **empty** DataArray (shape ``(0, 0, 0)``) when no files
        are available for the requested period or all conversions fail.

    Raises
    ------
    ValueError
        If *end* is before *start*, or if the reprojected polygon has zero
        area.

    Examples
    --------
    ::

        from shapely.geometry import box
        from eoflow.rainfall import get_rainfall_for_polygon

        # Small bounding box over Devon
        devon = box(-3.6, 50.6, -3.4, 50.8)
        da = get_rainfall_for_polygon(
            devon,
            start="2024-03-01T00:00",
            end="2024-03-01T01:00",
            download_dir="./rainfall_cache",
        )
        print(da)
        # <xarray.DataArray 'rainfall_rate' (time: 5, y: 24, x: 20)>
        # Coordinates:
        #   * time  (time) datetime64[ns] ...
        #   * y     (y)    float64  ...
        #   * x     (x)    float64  ...

        # Mean over the polygon for each timestep
        print(da.mean(dim=["y", "x"]))
    """
    import shutil
    import tempfile

    import shapely
    import xarray as xr

    if isinstance(start, str):
        start = parse_datetime(start)
    if isinstance(end, str):
        end = parse_datetime(end)

    if end < start:
        raise ValueError("'end' must be >= 'start'")

    # --- Reproject polygon and derive a pixel-aligned BNG extraction extent --
    polygon_bng = _polygon_to_bng(polygon)
    if polygon_bng.area == 0.0:
        raise ValueError("Input polygon has zero area after reprojection to BNG.")

    x_min_bb, y_min_bb, x_max_bb, y_max_bb = polygon_bng.bounds

    # Snap to the resolution grid so pixel centres are aligned consistently.
    x_min = float(np.floor(x_min_bb / res) * res)
    y_min = float(np.floor(y_min_bb / res) * res)
    x_max = float((np.ceil(x_max_bb / res) + 1) * res)
    y_max = float((np.ceil(y_max_bb / res) + 1) * res)

    logger.info("get_rainfall_for_polygon")
    logger.info(
        "  Polygon BNG bounds : %.0f, %.0f → %.0f, %.0f",
        x_min_bb,
        y_min_bb,
        x_max_bb,
        y_max_bb,
    )
    logger.info(
        "  Extraction extent  : x=[%.0f, %.0f]  y=[%.0f, %.0f]  res=%d m",
        x_min,
        x_max,
        y_min,
        y_max,
        res,
    )
    logger.info("  Time range         : %s → %s", start.isoformat(), end.isoformat())

    # --- Resolve download directory ------------------------------------------
    _tmp_dir: str | None = None
    if download_dir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="eoflow_rainfall_")
        dl_dir = Path(_tmp_dir)
        logger.debug("Using temporary download directory: %s", _tmp_dir)
    else:
        dl_dir = Path(download_dir)

    _empty_da = xr.DataArray(
        data=np.empty((0, 0, 0), dtype=np.float32),
        dims=["time", "y", "x"],
        name="rainfall_rate",
        attrs={"units": "mm/h", "crs": "EPSG:27700"},
    )

    try:
        # --- Step 1: Download NetCDF files ------------------------------------
        logger.info("Downloading rainfall NetCDF files ...")
        counts = download_rainfall(
            start=start,
            end=end,
            output_dir=dl_dir,
            run_hour=run_hour,
            workers=workers,
        )
        logger.info(
            "  Downloaded: %d  Skipped: %d  Errors: %d",
            counts["downloaded"],
            counts["skipped"],
            counts["error"],
        )

        # --- Step 2: Discover downloaded files --------------------------------
        nc_files = find_netcdf_files(dl_dir)
        if not nc_files:
            logger.warning("No rainfall NetCDF files found for the given time range.")
            return _empty_da

        logger.info("Regridding %d file(s) to BNG ...", len(nc_files))

        # --- Step 3: Pre-compute the BNG grid and polygon mask ---------------
        # The grid is determined entirely by the snapped extent and resolution,
        # so it is identical for every file — build the mask once.
        eastings = np.arange(x_min, x_max, res, dtype=np.float64)
        northings = np.arange(y_min, y_max, res, dtype=np.float64)

        xx, yy = np.meshgrid(eastings, northings)
        grid_points = shapely.points(xx.ravel(), yy.ravel())
        polygon_mask = shapely.within(grid_points, polygon_bng).reshape(xx.shape)

        n_inside = int(polygon_mask.sum())
        n_total = polygon_mask.size
        logger.info(
            "  Polygon covers %d / %d pixel(s) (%.1f%%) on the BNG grid.",
            n_inside,
            n_total,
            100.0 * n_inside / n_total if n_total > 0 else 0.0,
        )

        # --- Step 4: Load, reproject, and mask each file ----------------------
        slices: list[np.ndarray] = []
        valid_times: list[datetime] = []

        for nc_path in sorted(nc_files):
            # Derive the valid timestamp from the file path.
            _, valid_str = parse_timestamps_from_path(nc_path)
            vt: datetime | None = None
            if valid_str is not None:
                try:
                    vt = datetime.strptime(valid_str, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                except ValueError:
                    logger.warning("Could not parse valid time from path: %s", nc_path.name)

            try:
                data = _load_cube_for_polygon(nc_path, x_min, x_max, y_min, y_max, res)
            except Exception as exc:
                logger.error("  [error] failed to load %s: %s", nc_path.name, exc)
                continue

            # Mask cells outside the polygon.
            data[~polygon_mask] = np.nan

            slices.append(data)
            valid_times.append(vt)  # type: ignore[arg-type]
            logger.debug("  [ok] %s  (valid time: %s)", nc_path.name, vt)

        if not slices:
            logger.warning("No files were successfully processed.")
            return _empty_da

        # --- Step 5: Assemble xarray DataArray --------------------------------
        # Convert timezone-aware datetimes to timezone-naive numpy datetime64
        # so xarray can serialise the time coordinate to NetCDF.
        time_coords = np.array(
            [np.datetime64(t.replace(tzinfo=None), "ns") for t in valid_times],
            dtype="datetime64[ns]",
        )
        da = xr.DataArray(
            data=np.stack(slices, axis=0),  # (time, y, x)
            coords={
                "time": time_coords,
                "y": northings,
                "x": eastings,
            },
            dims=["time", "y", "x"],
            name="rainfall_rate",
            attrs={
                "units": "mm/h",
                "long_name": "Rainfall rate",
                "crs": "EPSG:27700",
                "source": "Met Office UKV 2km deterministic model",
            },
        )

        logger.info(
            "Assembled DataArray: %d timestep(s), grid %d (y) × %d (x).",
            len(valid_times),
            len(northings),
            len(eastings),
        )

        return da

    finally:
        if _tmp_dir is not None:
            shutil.rmtree(_tmp_dir, ignore_errors=True)
            logger.debug("Removed temporary download directory: %s", _tmp_dir)


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
    import os
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


def read_nimrod_file(dat_gz_path: Path) -> iris.cube.Cube | None:
    """Decompress and load a NIMROD ``.dat.gz`` file as an Iris cube.

    Parameters
    ----------
    dat_gz_path :
        Path to the compressed NIMROD binary file.

    Returns
    -------
    iris.cube.Cube or None
        Loaded cube, or ``None`` if reading fails.
    """
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as fh:
            tmp = Path(fh.name)
        with gzip.open(dat_gz_path, "rb") as gz, open(tmp, "wb") as out:
            shutil.copyfileobj(gz, out)
        return iris.load_cube(str(tmp))
    except Exception as exc:
        logger.error("Failed to read %s: %s", dat_gz_path, exc)
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
        iris.FUTURE.save_split_attrs = True
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iris.save(cube, str(output_path))
        return True
    except Exception as exc:
        logger.error("Failed to save %s: %s", output_path, exc)
        return False


# ---------------------------------------------------------------------------
# NIMROD 1km composite - consolidation (public)
# ---------------------------------------------------------------------------


def find_nimrod_nc_files(
    input_dir: Path,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[list[Path], int]:
    """Return sorted ``.nc`` paths under *input_dir*, optionally date-filtered.

    Files are expected to have stems of the form ``YYYYMMDD_HHMMSS``.  Any
    file whose stem cannot be parsed is always included.

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
        except ValueError:
            matched.append(f)
            continue
        if start is not None and dt < start:
            skipped += 1
            continue
        if end is not None and dt > end:
            skipped += 1
            continue
        matched.append(f)

    return matched, skipped


def consolidate_nimrod(
    input_dir: Path,
    output_path: Path,
    start: datetime | None = None,
    end: datetime | None = None,
    compression: int = 4,
    dry_run: bool = False,
) -> dict[str, int]:
    """Merge per-timestep NIMROD NetCDF files into a single NetCDF4 file.

    Each input file is a single-timestep ``.nc`` produced by
    :func:`process_nimrod_local` or :func:`download_and_crop_nimrod`.  Files
    are opened in parallel via Dask and merged along the time dimension,
    producing a dataset with shape
    ``(time, projection_y_coordinate, projection_x_coordinate)``.

    The key performance improvement over a naive ``open_mfdataset`` call is
    ``parallel=True``, which uses ``dask.delayed`` to open all files
    concurrently, and a ``preprocess`` step that promotes the scalar ``time``
    coordinate written by Iris into a proper 1-element dimension before
    concatenation.

    Parameters
    ----------
    input_dir :
        Root directory containing per-timestep ``.nc`` files (searched
        recursively).
    output_path :
        Destination ``.nc`` file to write.
    start :
        Optional inclusive start datetime for date-range filtering.
    end :
        Optional inclusive end datetime for date-range filtering.
    compression :
        zlib compression level 0-9 applied to all data variables (0 = none).
    dry_run :
        If ``True``, print which files would be merged without writing.

    Returns
    -------
    dict
        ``{"merged": int, "skipped": int}``
    """
    from tqdm.dask import TqdmCallback

    logger.info("Searching for NetCDF files in: %s", input_dir)
    nc_files, n_skipped = find_nimrod_nc_files(input_dir, start, end)

    if not nc_files:
        logger.error("No NetCDF files found in: %s", input_dir)
        return {"merged": 0, "skipped": n_skipped}

    logger.info("Found %d file(s) to merge (%d filtered out)", len(nc_files), n_skipped)

    if dry_run:
        logger.info("[dry-run] Would merge %d files -> %s", len(nc_files), output_path)
        for f in nc_files:
            try:
                rel = f.relative_to(input_dir)
            except ValueError:
                rel = f
            print(f"  {rel}")
        return {"merged": len(nc_files), "skipped": n_skipped}

    def _fix_time(ds: xr.Dataset) -> xr.Dataset:
        """Promote a scalar time coordinate to a 1-element dimension.

        Iris saves single-timestep cubes with time as a 0-d (scalar)
        coordinate.  xarray needs it to be a 1-element dimension so that
        ``concat_dim="time"`` can stack multiple files into one axis.
        """
        if "time" in ds.coords and ds["time"].ndim == 0:
            return ds.expand_dims("time")
        return ds

    logger.info("Opening %d file(s) with xarray (parallel=True) ...", len(nc_files))
    try:
        ds = xr.open_mfdataset(
            [str(f) for f in nc_files],
            # parallel=True uses dask.delayed to open every file
            # concurrently — a significant speedup for thousands of files.
            parallel=True,
            # preprocess promotes the scalar time coordinate written by Iris
            # into a 1-element dimension before concatenation.
            preprocess=_fix_time,
            combine="nested",
            concat_dim="time",
            # 288 five-minute steps = one day; a natural chunk boundary.
            chunks={"time": 288},
            compat="override",
            coords="minimal",
        )
    except Exception as exc:
        logger.error("Failed to open/merge files: %s", exc, exc_info=True)
        raise

    time_start = str(ds.time.values[0])[:19]
    time_end = str(ds.time.values[-1])[:19]
    logger.info("  Dimensions : %s", dict(ds.sizes))
    logger.info("  Variables  : %s", list(ds.data_vars))
    logger.info("  Time range : %s -> %s", time_start, time_end)
    logger.info("  Timesteps  : %d", ds.sizes.get("time", "?"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {var: {"zlib": compression > 0, "complevel": compression} for var in ds.data_vars}

    logger.info("Writing merged dataset to: %s", output_path)
    with TqdmCallback(desc="Writing", unit="chunk"):
        ds.to_netcdf(output_path, format="NETCDF4", encoding=encoding)

    size_mb = output_path.stat().st_size / (1024**2)
    logger.info("Done.  Output: %s (%.1f MB)", output_path, size_mb)

    return {"merged": len(nc_files), "skipped": n_skipped}


# ---------------------------------------------------------------------------
# NIMROD 1km composite - local tar processing (public)
# ---------------------------------------------------------------------------


def process_nimrod_tar(
    tar_path: Path,
    gdf: gpd.GeoDataFrame,
    output_dir: Path,
    temp_dir: Path,
) -> dict[str, int]:
    """Extract and process one locally downloaded NIMROD tar file.

    Reads each ``.dat.gz`` timestep inside the tar with Iris, crops it to
    *gdf*, and writes per-timestep NetCDF files under
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
        Scratch directory for extraction; cleaned up on return.

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

    extract_dir = temp_dir / date_str
    dat_gz_files = extract_nimrod_tar(tar_path, extract_dir)

    if not dat_gz_files:
        logger.error("  [error] No .dat.gz files found in %s", tar_path.name)
        return {"processed": 0, "errors": 1}

    processed = error_count = 0

    for dat_gz_path in dat_gz_files:
        cube = read_nimrod_file(dat_gz_path)
        if cube is None:
            error_count += 1
            continue

        cropped = crop_nimrod_cube(cube, gdf)
        if cropped is None:
            error_count += 1
            continue

        try:
            time_coord = cropped.coord("time")
            ts = time_coord.units.num2date(time_coord.points[0])
            timestamp_str = ts.strftime("%Y%m%d_%H%M%S")
        except Exception:
            timestamp_str = dat_gz_path.stem.replace(".dat", "")

        if save_nimrod_cube(cropped, date_output_dir / f"{timestamp_str}.nc"):
            processed += 1
        else:
            error_count += 1

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    if error_count == 0:
        logger.info("  [done] Processed %d timestep(s)", processed)
    else:
        logger.warning("  [done] %d timestep(s) processed, %d error(s)", processed, error_count)

    return {"processed": processed, "errors": error_count}


def process_nimrod_local(
    input_paths: list[Path],
    shapefile_path: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Process locally downloaded NIMROD tar files and crop to a shapefile.

    Discovers ``.tar`` files in *input_paths* (files or directories),
    extracts each one, reads every ``.dat.gz`` timestep with Iris, crops
    it to *shapefile_path*, and writes per-timestep NetCDF files under
    ``output_dir/<year>/<YYYYMMDD>/``.

    Parameters
    ----------
    input_paths :
        List of ``.tar`` file paths or directories containing ``.tar`` files.
    shapefile_path :
        Path to the shapefile used for spatial cropping (any input CRS).
    output_dir :
        Root directory for processed NetCDF files.

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

    logger.info("Found %d tar file(s) to process", len(tar_files))
    results: dict[str, int] = {"processed": 0, "errors": 0}

    for tar_path in sorted(tar_files):
        r = process_nimrod_tar(tar_path, gdf, output_dir, temp_dir)
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

    extract_dir = temp_dir / date_str
    dat_gz_files = extract_nimrod_tar(tar_path, extract_dir)

    if not dat_gz_files:
        with print_lock:
            logger.error("  [error] No .dat.gz files in tar for %s", date_str)
        tar_path.unlink(missing_ok=True)
        return {"processed": 0, "skipped": 0, "errors": 1, "dry_run": 0}

    processed = error_count = 0

    for dat_gz_path in dat_gz_files:
        cube = read_nimrod_file(dat_gz_path)
        if cube is None:
            error_count += 1
            continue

        cropped = crop_nimrod_cube(cube, gdf)
        if cropped is None:
            error_count += 1
            continue

        try:
            time_coord = cropped.coord("time")
            ts = time_coord.units.num2date(time_coord.points[0])
            timestamp_str = ts.strftime("%Y%m%d_%H%M%S")
        except Exception:
            timestamp_str = dat_gz_path.stem.replace(".dat", "")

        if save_nimrod_cube(cropped, date_output_dir / f"{timestamp_str}.nc"):
            processed += 1
        else:
            error_count += 1

    tar_path.unlink(missing_ok=True)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

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


# ---------------------------------------------------------------------------
# Convenience re-export
# ---------------------------------------------------------------------------

__all__ = [
    # Download
    "download_rainfall",
    "generate_valid_times",
    "parse_datetime",
    "candidate_run_times",
    "find_s3_key_for_valid_time",
    "make_s3_client",
    "s3_key",
    # Convert
    "convert_rainfall",
    "convert_netcdf_to_asc",
    "find_netcdf_files",
    "output_path_for",
    "parse_timestamps_from_path",
    "resolve_extent",
    # Constants
    "BUCKET",
    "PREFIX",
    "NC_SUFFIX",
    "TIMESTEP_MINUTES",
    "MAX_LEAD_HOURS",
    "DEFAULT_RESOLUTION_M",
    "UK_BNG_EXTENT",
    "ALL_RUN_HOURS",
]
