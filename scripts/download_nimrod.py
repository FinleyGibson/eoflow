"""
Download Met Office NIMROD 1km composite rainfall data from CEDA and crop to a shapefile.

This script downloads NIMROD radar rainfall data from the CEDA archive, extracts
the individual timestep files from the daily tar archives, reads them with Iris,
and crops them to a user-provided shapefile boundary.

Data Source
-----------
``https://data.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km/``

Features
--------
- Downloads data for specified year(s)
- Parallel download and processing with configurable workers
- Automatic extraction of `.dat.gz.tar` files
- Cropping to shapefile boundary
- Skips already processed files
- Supports dry-run mode for testing

File Structure
--------------
CEDA organizes NIMROD data by year, with one tar file per day::

    /{year}/metoffice-c-band-rain-radar_uk_{YYYYMMDD}_1km-composite.dat.gz.tar

Each tar file contains multiple timesteps (typically 5-minute intervals).

Usage
-----
::

    # Download and crop 2024 data to a Devon shapefile
    python -m scripts.download_nimrod \\
        --years 2024 \\
        --shapefile ./data/devon.shp \\
        --output ./nimrod_data \\
        --workers 4

    # Download multiple years
    python -m scripts.download_nimrod \\
        --years 2023 2024 \\
        --shapefile ./data/catchment.shp \\
        --output ./nimrod_data

    # Dry-run to see what would be downloaded
    python -m scripts.download_nimrod \\
        --years 2024 \\
        --shapefile ./data/devon.shp \\
        --dry-run

    # Download specific date range within a year
    python -m scripts.download_nimrod \\
        --years 2024 \\
        --start-date 2024-01-01 \\
        --end-date 2024-01-31 \\
        --shapefile ./data/devon.shp \\
        --output ./nimrod_data

Output Format
-------------
Cropped data is saved as NetCDF files in the output directory, organized by date::

    <output_dir>/{year}/{YYYYMMDD}/{timestamp}.nc

Each NetCDF file contains rainfall rate data cropped to the shapefile extent,
with coordinates in British National Grid (EPSG:27700).

Requirements
------------
- Valid CEDA account credentials (if required for access)
- Shapefile in any CRS (will be reprojected to BNG if needed)
- Sufficient disk space for temporary tar files and output
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import rasterio.features
import rasterio.transform
import requests

from eoflow.log_utils import get_logger

if TYPE_CHECKING:
    import iris
    import iris.cube
else:
    import iris
    import iris.cube

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CEDA_BASE_URL = "https://dap.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km"
FILE_PATTERN = "metoffice-c-band-rain-radar_uk_{date}_1km-composite.dat.gz.tar"

# British National Grid (the native CRS of NIMROD data)
BNG_EPSG = 27700


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------


def generate_dates(
    start_date: date,
    end_date: date,
) -> list[date]:
    """Generate list of dates between start and end (inclusive).

    Parameters
    ----------
    start_date :
        First date to include.
    end_date :
        Last date to include.

    Returns
    -------
    list[date]
        All dates from start_date to end_date inclusive.
    """
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def parse_date_string(date_str: str) -> date:
    """Parse YYYY-MM-DD date string.

    Parameters
    ----------
    date_str :
        Date in YYYY-MM-DD format.

    Returns
    -------
    date
        Parsed date object.

    Raises
    ------
    ValueError
        If date string is invalid.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date format '{date_str}', expected YYYY-MM-DD") from exc


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------


def create_session() -> requests.Session:
    """Create a requests session with CEDA authentication support.

    Checks for credentials in this order:
    1. CEDA_TOKEN environment variable (OAuth token)
    2. ~/.netrc file (machine dap.ceda.ac.uk)
    3. Environment variables: CEDA_USERNAME and CEDA_PASSWORD

    Returns
    -------
    requests.Session
        Configured session object.
    """
    import os
    from netrc import NetrcParseError, netrc

    session = requests.Session()

    # Try OAuth token first (most reliable)
    token = os.environ.get("CEDA_TOKEN")
    if token:
        # Try different token authentication methods CEDA might use:
        # 1. Bearer token in Authorization header
        session.headers.update({"Authorization": f"Bearer {token}"})
        logger.info("Using CEDA OAuth token from environment variable (Bearer)")
        return session

    # Try to get credentials from .netrc
    try:
        rc = netrc()
        auth_info = rc.authenticators("dap.ceda.ac.uk")
        if auth_info:
            username, _, password = auth_info
            session.auth = (username, password)
            logger.info("Using CEDA credentials from .netrc for user: %s", username)
            return session
    except (FileNotFoundError, NetrcParseError, TypeError):
        pass  # .netrc doesn't exist or is invalid

    # Try environment variables
    username = os.environ.get("CEDA_USERNAME")
    password = os.environ.get("CEDA_PASSWORD")

    if username and password:
        session.auth = (username, password)
        logger.info("Using CEDA credentials from environment variables for user: %s", username)
        return session

    logger.warning(
        "No CEDA credentials found. Downloads may fail if authentication is required. "
        "Set up authentication using: "
        "1) CEDA_TOKEN environment variable (recommended), "
        "2) ~/.netrc file, or "
        "3) CEDA_USERNAME/CEDA_PASSWORD environment variables."
    )

    return session


def build_file_url(year: int, date_obj: date) -> str:
    """Build CEDA URL for a given date.

    Parameters
    ----------
    year :
        Year directory.
    date_obj :
        Date for which to build URL.

    Returns
    -------
    str
        Full URL to the tar file.
    """
    date_str = date_obj.strftime("%Y%m%d")
    filename = FILE_PATTERN.format(date=date_str)
    return f"{CEDA_BASE_URL}/{year}/{filename}"


def download_file(
    url: str,
    output_path: Path,
    session: requests.Session | None = None,
    timeout: int = 300,
    chunk_size: int = 8192,
) -> bool:
    """Download a file from URL to output_path.

    Parameters
    ----------
    url :
        URL to download from.
    output_path :
        Local path to save file.
    session :
        Optional requests session with authentication configured.
        If None, creates a new session.
    timeout :
        Request timeout in seconds.
    chunk_size :
        Download chunk size in bytes.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    if session is None:
        session = requests.Session()

    try:
        # Make request - allow redirects but check for auth redirects
        response = session.get(url, stream=True, timeout=timeout, allow_redirects=True)

        # Check if we got redirected to authentication page
        if "auth.ceda.ac.uk" in response.url or "signin" in response.url.lower():
            logger.error(
                "Authentication required for %s. "
                "This dataset requires CEDA credentials. "
                "Please set up authentication (see README for details).",
                url,
            )
            return False

        response.raise_for_status()

        # Check content type - should be application/x-tar or similar
        content_type = response.headers.get("content-type", "").lower()
        if "html" in content_type:
            logger.error(
                "Received HTML instead of tar file from %s. "
                "This may indicate an authentication issue or server error. "
                "Content-Type: %s",
                url,
                content_type,
            )
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress tracking
        downloaded_size = 0
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

        # Validate downloaded file
        if downloaded_size == 0:
            logger.error("Downloaded file is empty: %s", url)
            output_path.unlink(missing_ok=True)
            return False

        # Check if it looks like a tar file (starts with tar magic bytes or is valid)
        try:
            with open(output_path, "rb") as f:
                header = f.read(512)  # Read tar header
                # Basic validation - tar files have specific structure
                if len(header) < 512:
                    logger.error(
                        "Downloaded file too small (%d bytes), expected tar file: %s",
                        downloaded_size,
                        url,
                    )
                    return False

                # Check if it's HTML (common error case)
                if header[:15].lower() == b"<!doctype html" or header[:6].lower() == b"<html":
                    logger.error(
                        "Downloaded file is HTML, not a tar file. "
                        "This usually indicates an authentication or access issue. "
                        "URL: %s",
                        url,
                    )
                    output_path.unlink(missing_ok=True)
                    return False

        except Exception as read_exc:
            logger.error("Failed to validate downloaded file: %s", read_exc)
            return False

        logger.debug("Downloaded %s (%d bytes)", output_path.name, downloaded_size)
        return True

    except requests.exceptions.RequestException as exc:
        logger.error("Failed to download %s: %s", url, exc)
        output_path.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Processing utilities
# ---------------------------------------------------------------------------


def load_shapefile(shapefile_path: Path) -> gpd.GeoDataFrame:
    """Load shapefile and reproject to BNG if needed.

    Parameters
    ----------
    shapefile_path :
        Path to shapefile.

    Returns
    -------
    GeoDataFrame
        Shapefile data in BNG (EPSG:27700).
    """
    gdf = gpd.read_file(shapefile_path)

    if gdf.crs is None:
        logger.warning("Shapefile has no CRS defined, assuming BNG (EPSG:27700)")
        gdf.set_crs(epsg=BNG_EPSG, inplace=True)
    elif gdf.crs.to_epsg() != BNG_EPSG:
        logger.info("Reprojecting shapefile from %s to BNG", gdf.crs)
        gdf = gdf.to_crs(epsg=BNG_EPSG)

    return gdf


def extract_tar_file(
    tar_path: Path,
    extract_dir: Path,
) -> list[Path]:
    """Extract a tar file and return list of extracted .dat.gz files.

    Parameters
    ----------
    tar_path :
        Path to tar file.
    extract_dir :
        Directory to extract to.

    Returns
    -------
    list[Path]
        List of extracted .dat.gz file paths.
    """
    extracted_files = []

    try:
        # Try different modes - the CEDA files are uncompressed tar files
        # despite having .tar extension (not .tar.gz)
        tar_opened = False

        for mode in ["r:", "r", "r:gz", "r:bz2", "r:xz"]:
            try:
                with tarfile.open(tar_path, mode) as tar:  # type: ignore[arg-type]
                    tar_opened = True
                    extract_dir.mkdir(parents=True, exist_ok=True)

                    # List all members for debugging
                    members = tar.getmembers()
                    logger.debug(
                        "Tar file %s contains %d members (mode: %s)",
                        tar_path.name,
                        len(members),
                        mode,
                    )

                    # Log first few filenames for debugging
                    if members:
                        sample_names = [m.name for m in members[:5]]
                        logger.debug("Sample filenames: %s", sample_names)

                    tar.extractall(path=extract_dir)

                    for member in members:
                        if member.name.endswith(".dat.gz"):
                            extracted_path = extract_dir / member.name
                            extracted_files.append(extracted_path)

                    if not extracted_files:
                        # Log all filenames if no .dat.gz found
                        all_names = [m.name for m in members]
                        logger.warning(
                            "No .dat.gz files found in tar. All members: %s",
                            all_names[:20],  # First 20 to avoid spam
                        )

                    break  # Successfully opened and extracted
            except (tarfile.ReadError, tarfile.CompressionError) as tar_exc:
                logger.debug("Mode %s failed: %s", mode, tar_exc)
                continue  # Try next mode

        if not tar_opened:
            logger.error("Failed to open %s with any tar mode", tar_path)
            # Try to inspect the file
            try:
                file_size = tar_path.stat().st_size
                with open(tar_path, "rb") as f:
                    first_bytes = f.read(512)
                logger.error(
                    "File size: %d bytes. First 64 bytes (hex): %s",
                    file_size,
                    first_bytes[:64].hex(),
                )
            except Exception:
                pass

    except (tarfile.TarError, OSError) as exc:
        logger.error("Failed to extract %s: %s", tar_path, exc)

    return extracted_files


def read_nimrod_file(dat_gz_path: Path) -> iris.cube.Cube | None:  # type: ignore[name-defined]
    """Read a NIMROD .dat.gz file using Iris.

    Parameters
    ----------
    dat_gz_path :
        Path to .dat.gz file.

    Returns
    -------
    iris.cube.Cube or None
        Iris cube if successful, None otherwise.
    """
    tmp_dat_path = None
    try:
        # Decompress .gz to temporary .dat file
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp_dat:
            tmp_dat_path = Path(tmp_dat.name)

        with gzip.open(dat_gz_path, "rb") as f_in:
            with open(tmp_dat_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Load with Iris
        cube = iris.load_cube(str(tmp_dat_path))

        # Clean up temporary file
        tmp_dat_path.unlink()

        return cube

    except Exception as exc:
        logger.error("Failed to read %s: %s", dat_gz_path, exc)
        if tmp_dat_path is not None and tmp_dat_path.exists():
            tmp_dat_path.unlink()
        return None


def crop_cube_to_shapefile(
    cube: iris.cube.Cube,  # type: ignore[name-defined]
    gdf: gpd.GeoDataFrame,
) -> iris.cube.Cube | None:  # type: ignore[name-defined]
    """Crop an Iris cube to shapefile extent and mask outside polygons.

    Parameters
    ----------
    cube :
        Iris cube with BNG coordinates.
    gdf :
        GeoDataFrame in BNG with polygons to crop to.

    Returns
    -------
    iris.cube.Cube or None
        Cropped cube with masked data outside polygons.
    """
    try:
        # Get cube coordinates
        x_coord = cube.coord(axis="X")
        y_coord = cube.coord(axis="Y")

        # Determine if coords are 1D or 2D
        if x_coord.ndim == 1 and y_coord.ndim == 1:
            # Regular grid
            x_vals = x_coord.points
            y_vals = y_coord.points

            # Get bounds from shapefile
            minx, miny, maxx, maxy = gdf.total_bounds

            # Find indices within bounds
            x_mask = (x_vals >= minx) & (x_vals <= maxx)
            y_mask = (y_vals >= miny) & (y_vals <= maxy)

            x_indices = np.where(x_mask)[0]
            y_indices = np.where(y_mask)[0]

            if len(x_indices) == 0 or len(y_indices) == 0:
                logger.warning("Shapefile bounds do not intersect with data extent")
                return None

            # Slice the cube
            x_slice = slice(x_indices[0], x_indices[-1] + 1)
            y_slice = slice(y_indices[0], y_indices[-1] + 1)

            # Determine which dimension is x and which is y
            x_dim = cube.coord_dims(x_coord)[0]
            y_dim = cube.coord_dims(y_coord)[0]

            if x_dim < y_dim:
                cropped = cube[:, x_slice, y_slice] if cube.ndim == 3 else cube[x_slice, y_slice]
            else:
                cropped = cube[:, y_slice, x_slice] if cube.ndim == 3 else cube[y_slice, x_slice]

            # Create mask for areas outside polygons
            x_crop = cropped.coord(axis="X").points
            y_crop = cropped.coord(axis="Y").points

            # Build affine transform
            x_res = np.median(np.diff(x_crop))
            y_res = np.median(np.diff(y_crop))

            # Use abs() for half-pixel padding so the bounds are correct
            # regardless of whether y_res is positive or negative.
            transform = rasterio.transform.from_bounds(
                x_crop.min() - abs(x_res) / 2,
                y_crop.min() - abs(y_res) / 2,
                x_crop.max() + abs(x_res) / 2,
                y_crop.max() + abs(y_res) / 2,
                len(x_crop),
                len(y_crop),
            )

            # Create geometry mask (True where outside polygons)
            geometries = [geom for geom in gdf.geometry]
            mask = rasterio.features.geometry_mask(
                geometries,
                out_shape=(len(y_crop), len(x_crop)),
                transform=transform,
                invert=False,
            )

            # rasterio.transform.from_bounds produces a north-up affine
            # (row 0 = north).  If the cube's y coordinate is ascending
            # (row 0 = south), flip the mask so it matches the data.
            if len(y_crop) > 1 and float(y_crop[0]) < float(y_crop[-1]):
                mask = mask[::-1, :]

            # Apply mask to data
            data = cropped.data
            if hasattr(data, "mask"):
                # Already a masked array
                data.mask = data.mask | mask  # type: ignore[union-attr]
            else:
                # Convert to masked array
                data = np.ma.masked_array(data, mask=mask)

            cropped.data = data

            return cropped

        else:
            logger.warning("Non-regular grids not yet supported")
            return None

    except Exception as exc:
        logger.error("Failed to crop cube: %s", exc)
        return None


def save_cube_as_netcdf(
    cube: iris.cube.Cube,  # type: ignore[name-defined]
    output_path: Path,
) -> bool:
    """Save Iris cube as NetCDF file.

    Parameters
    ----------
    cube :
        Iris cube to save.
    output_path :
        Output NetCDF path.

    Returns
    -------
    bool
        True if successful, False otherwise.
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
# Main processing pipeline
# ---------------------------------------------------------------------------


def process_date(
    date_obj: date,
    year: int,
    gdf: gpd.GeoDataFrame,
    output_dir: Path,
    temp_dir: Path,
    dry_run: bool,
    print_lock: threading.Lock,
    session: requests.Session,
) -> dict[str, int]:
    """Download and process NIMROD data for a single date.

    Parameters
    ----------
    date_obj :
        Date to process.
    year :
        Year (for URL construction).
    gdf :
        Shapefile geodataframe in BNG.
    output_dir :
        Output directory for NetCDF files.
    temp_dir :
        Temporary directory for downloads.
    dry_run :
        If True, don't actually download or process.
    print_lock :
        Thread lock for logging.
    session :
        Requests session with authentication.

    Returns
    -------
    dict
        Counts of processed, skipped, and errors.
    """
    date_str = date_obj.strftime("%Y%m%d")
    url = build_file_url(year, date_obj)

    with print_lock:
        logger.info("Processing %s", date_str)

    if dry_run:
        with print_lock:
            logger.info("  [dry-run] Would download: %s", url)
        return {"processed": 0, "skipped": 0, "errors": 0, "dry_run": 1}

    # Check if already processed
    date_output_dir = output_dir / str(year) / date_str
    if date_output_dir.exists() and any(date_output_dir.glob("*.nc")):
        with print_lock:
            logger.info("  [skip] Already processed")
        return {"processed": 0, "skipped": 1, "errors": 0, "dry_run": 0}

    # Download tar file
    tar_path = temp_dir / f"{date_str}.tar"
    if not download_file(url, tar_path, session=session):
        with print_lock:
            logger.error("  [error] Download failed")
        return {"processed": 0, "skipped": 0, "errors": 1, "dry_run": 0}

    # Extract tar file
    extract_dir = temp_dir / date_str
    dat_gz_files = extract_tar_file(tar_path, extract_dir)

    if not dat_gz_files:
        with print_lock:
            logger.error("  [error] No .dat.gz files found in tar")
        tar_path.unlink(missing_ok=True)
        return {"processed": 0, "skipped": 0, "errors": 1, "dry_run": 0}

    # Process each timestep file
    processed_count = 0
    error_count = 0

    for dat_gz_path in dat_gz_files:
        # Read NIMROD file
        cube = read_nimrod_file(dat_gz_path)
        if cube is None:
            error_count += 1
            continue

        # Crop to shapefile
        cropped_cube = crop_cube_to_shapefile(cube, gdf)
        if cropped_cube is None:
            error_count += 1
            continue

        # Generate output filename from cube time
        try:
            time_coord = cropped_cube.coord("time")
            timestamp = time_coord.units.num2date(time_coord.points[0])
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        except Exception:
            # Fallback to filename
            timestamp_str = dat_gz_path.stem.replace(".dat", "")

        output_path = date_output_dir / f"{timestamp_str}.nc"

        # Save as NetCDF
        if save_cube_as_netcdf(cropped_cube, output_path):
            processed_count += 1
        else:
            error_count += 1

    # Clean up temporary files
    tar_path.unlink(missing_ok=True)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    with print_lock:
        if error_count == 0:
            logger.info("  [done] Processed %d timesteps", processed_count)
        else:
            logger.warning(
                "  [done] Processed %d timesteps with %d errors",
                processed_count,
                error_count,
            )

    return {
        "processed": processed_count,
        "skipped": 0,
        "errors": error_count,
        "dry_run": 0,
    }


def download_and_crop_nimrod(
    years: list[int],
    shapefile_path: Path,
    output_dir: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    workers: int = 4,
    dry_run: bool = False,
) -> dict[str, int]:
    """Download NIMROD data for specified years and crop to shapefile.

    Parameters
    ----------
    years :
        List of years to download.
    shapefile_path :
        Path to shapefile for cropping.
    output_dir :
        Output directory for processed NetCDF files.
    start_date :
        Optional start date filter (inclusive).
    end_date :
        Optional end date filter (inclusive).
    workers :
        Number of parallel workers.
    dry_run :
        If True, only show what would be downloaded.

    Returns
    -------
    dict
        Counts of processed, skipped, errors, and dry-run items.
    """
    logger.info("NIMROD 1km composite downloader and cropper")
    logger.info("  Years      : %s", years)
    logger.info("  Shapefile  : %s", shapefile_path)
    logger.info("  Output dir : %s", output_dir)
    logger.info("  Workers    : %d", workers)
    logger.info("  Dry run    : %s", dry_run)

    # Load shapefile
    logger.info("Loading shapefile...")
    gdf = load_shapefile(shapefile_path)
    logger.info("  Bounds (BNG): %s", gdf.total_bounds)

    # Create output and temp directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)

    # Generate list of dates to process
    dates_to_process = []
    for year in years:
        year_start = start_date if start_date and start_date.year == year else date(year, 1, 1)
        year_end = end_date if end_date and end_date.year == year else date(year, 12, 31)
        dates_to_process.extend(generate_dates(year_start, year_end))

    logger.info("Processing %d dates across %d year(s)", len(dates_to_process), len(years))

    # Create authenticated session for downloads
    session = create_session()

    # Process dates in parallel
    print_lock = threading.Lock()
    results: dict[str, int] = {"processed": 0, "skipped": 0, "errors": 0, "dry_run": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_date,
                date_obj,
                date_obj.year,
                gdf,
                output_dir,
                temp_dir,
                dry_run,
                print_lock,
                session,  # Pass session to each worker
            ): date_obj
            for date_obj in dates_to_process
        }

        for future in as_completed(futures):
            date_result = future.result()
            for key, value in date_result.items():
                results[key] += value

    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    logger.info("Done.")
    logger.info("  Files processed : %d", results["processed"])
    logger.info("  Dates skipped   : %d", results["skipped"])
    logger.info("  Errors          : %d", results["errors"])
    if dry_run:
        logger.info("  Dry-run items   : %d", results["dry_run"])

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Download Met Office NIMROD 1km composite rainfall data from CEDA "
            "and crop to a shapefile."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        metavar="YEAR",
        help="Year(s) to download (e.g., 2024 or 2023 2024).",
    )
    p.add_argument(
        "--shapefile",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to shapefile for cropping (any CRS, will be reprojected to BNG).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("./nimrod_data"),
        metavar="DIR",
        help="Output directory for processed NetCDF files (default: %(default)s).",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Optional start date filter (inclusive).",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Optional end date filter (inclusive).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers (default: %(default)s).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # Validate dates if provided
    start_date = None
    end_date = None

    if args.start_date:
        try:
            start_date = parse_date_string(args.start_date)
        except ValueError as exc:
            logger.error("Invalid --start-date: %s", exc)
            sys.exit(1)

    if args.end_date:
        try:
            end_date = parse_date_string(args.end_date)
        except ValueError as exc:
            logger.error("Invalid --end-date: %s", exc)
            sys.exit(1)

    if start_date and end_date and end_date < start_date:
        logger.error("--end-date must be >= --start-date")
        sys.exit(1)

    # Validate shapefile exists
    if not args.shapefile.exists():
        logger.error("Shapefile not found: %s", args.shapefile)
        sys.exit(1)

    try:
        results = download_and_crop_nimrod(
            years=args.years,
            shapefile_path=args.shapefile,
            output_dir=args.output,
            start_date=start_date,
            end_date=end_date,
            workers=args.workers,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Exit with error if everything failed
    if results["errors"] > 0 and results["processed"] == 0 and results["skipped"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
