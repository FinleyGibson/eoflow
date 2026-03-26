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

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import iris
import iris.analysis
import iris.coord_systems
import iris.coords
import iris.cube
import numpy as np
import rasterio
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
) -> "xarray.DataArray":
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
