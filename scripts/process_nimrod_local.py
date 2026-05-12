"""
Process locally downloaded NIMROD tar files and crop to shapefile.

This script processes NIMROD data that has been manually downloaded from CEDA,
extracting the tar files and cropping to a shapefile boundary.

Usage
-----
::

    # Process files in a directory
    python -m scripts.process_nimrod_local \\
        --input ./downloaded_nimrod \\
        --shapefile ./data/devon.shp \\
        --output ./nimrod_data

    # Process specific tar files
    python -m scripts.process_nimrod_local \\
        --files file1.tar file2.tar \\
        --shapefile ./data/devon.shp \\
        --output ./nimrod_data
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from eoflow.log_utils import get_logger

# Import functions from download_nimrod
from scripts.download_nimrod import (
    crop_cube_to_shapefile,
    extract_tar_file,
    load_shapefile,
    read_nimrod_file,
    save_cube_as_netcdf,
)

logger = get_logger(__name__)


def process_local_tar_file(
    tar_path: Path,
    gdf,
    output_dir: Path,
    temp_dir: Path,
) -> dict[str, int]:
    """Process a locally downloaded tar file.

    Parameters
    ----------
    tar_path :
        Path to tar file.
    gdf :
        Shapefile GeoDataFrame in BNG.
    output_dir :
        Output directory for NetCDF files.
    temp_dir :
        Temporary directory for extraction.

    Returns
    -------
    dict
        Counts of processed and errors.
    """
    logger.info("Processing %s", tar_path.name)

    # Parse date from filename
    # Format: metoffice-c-band-rain-radar_uk_YYYYMMDD_1km-composite.dat.gz.tar
    try:
        # Remove .tar extension first, then split
        filename_without_tar = tar_path.stem  # Removes .tar
        # Split by underscore and get the date part (index 2)
        # metoffice-c-band-rain-radar_uk_YYYYMMDD_1km-composite.dat.gz
        parts = filename_without_tar.split("_")
        date_str = parts[2]  # YYYYMMDD
        year = date_str[:4]
    except (IndexError, ValueError) as e:
        logger.error("Could not parse date from filename: %s (error: %s)", tar_path.name, e)
        return {"processed": 0, "errors": 1}

    # Check if already processed
    date_output_dir = output_dir / year / date_str
    if date_output_dir.exists() and any(date_output_dir.glob("*.nc")):
        logger.info("  [skip] Already processed")
        return {"processed": 0, "errors": 0}

    # Extract tar file
    extract_dir = temp_dir / date_str
    dat_gz_files = extract_tar_file(tar_path, extract_dir)

    if not dat_gz_files:
        logger.error("  [error] No .dat.gz files found in tar")
        return {"processed": 0, "errors": 1}

    # Process each timestep
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

        # Generate output filename
        try:
            time_coord = cropped_cube.coord("time")
            timestamp = time_coord.units.num2date(time_coord.points[0])
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        except Exception:
            timestamp_str = dat_gz_path.stem.replace(".dat", "")

        output_path = date_output_dir / f"{timestamp_str}.nc"

        # Save as NetCDF
        if save_cube_as_netcdf(cropped_cube, output_path):
            processed_count += 1
        else:
            error_count += 1

    # Clean up
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    if error_count == 0:
        logger.info("  [done] Processed %d timesteps", processed_count)
    else:
        logger.warning(
            "  [done] Processed %d timesteps with %d errors",
            processed_count,
            error_count,
        )

    return {"processed": processed_count, "errors": error_count}


def process_local_nimrod(
    input_paths: list[Path],
    shapefile_path: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Process locally downloaded NIMROD files.

    Parameters
    ----------
    input_paths :
        List of tar file paths or directories containing tar files.
    shapefile_path :
        Path to shapefile for cropping.
    output_dir :
        Output directory for processed NetCDF files.

    Returns
    -------
    dict
        Counts of processed and errors.
    """
    logger.info("NIMROD local file processor")
    logger.info("  Shapefile  : %s", shapefile_path)
    logger.info("  Output dir : %s", output_dir)

    # Load shapefile
    logger.info("Loading shapefile...")
    gdf = load_shapefile(shapefile_path)
    logger.info("  Bounds (BNG): %s", gdf.total_bounds)

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)

    # Find all tar files
    tar_files = []
    for path in input_paths:
        if path.is_file() and path.suffix == ".tar":
            tar_files.append(path)
        elif path.is_dir():
            tar_files.extend(path.glob("*.tar"))

    if not tar_files:
        logger.error("No tar files found in: %s", input_paths)
        return {"processed": 0, "errors": 0}

    logger.info("Found %d tar files to process", len(tar_files))

    # Process each file
    results = {"processed": 0, "errors": 0}

    for tar_path in sorted(tar_files):
        file_results = process_local_tar_file(tar_path, gdf, output_dir, temp_dir)
        results["processed"] += file_results["processed"]
        results["errors"] += file_results["errors"]

    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    logger.info("Done.")
    logger.info("  Files processed : %d", results["processed"])
    logger.info("  Errors          : %d", results["errors"])

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Process locally downloaded NIMROD tar files and crop to shapefile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=Path,
        metavar="DIR",
        help="Directory containing downloaded tar files.",
    )
    input_group.add_argument(
        "--files",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Specific tar files to process.",
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

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # Validate shapefile exists
    if not args.shapefile.exists():
        logger.error("Shapefile not found: %s", args.shapefile)
        sys.exit(1)

    # Get input paths
    if args.input:
        if not args.input.exists():
            logger.error("Input directory not found: %s", args.input)
            sys.exit(1)
        input_paths = [args.input]
    else:
        # Validate all files exist
        for f in args.files:
            if not f.exists():
                logger.error("File not found: %s", f)
                sys.exit(1)
        input_paths = args.files

    try:
        results = process_local_nimrod(
            input_paths=input_paths,
            shapefile_path=args.shapefile,
            output_dir=args.output,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Exit with error if everything failed
    if results["errors"] > 0 and results["processed"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
