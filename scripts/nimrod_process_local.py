"""
Process locally downloaded NIMROD tar files and crop to shapefile.

This script is a thin CLI wrapper around
:func:`eoflow.rainfall.process_nimrod_local`.

The NIMROD tar files are expected to have been downloaded from CEDA using
``nimrod_download_script.sh`` or an equivalent tool.

Usage
-----
::

    # Process all tar files in a directory
    python -m scripts.process_nimrod_local \\
        --input ./downloaded_nimrod \\
        --shapefile ./data/devon.shp \\
        --output ./nimrod_data

    # Process specific tar files
    python -m scripts.process_nimrod_local \\
        --files file1.tar file2.tar \\
        --shapefile ./data/devon.shp \\
        --output ./nimrod_data

Output Format
-------------
Per-timestep NetCDF files organised by date::

    <output_dir>/{year}/{YYYYMMDD}/{YYYYMMDD_HHMMSS}.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eoflow.log_utils import get_logger
from eoflow.rainfall import process_nimrod_local

logger = get_logger(__file__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        help="Path to shapefile for cropping (any CRS, reprojected to BNG automatically).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("./nimrod_data"),
        metavar="DIR",
        help="Output directory for processed NetCDF files (default: %(default)s).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel worker threads (default: one per CPU core).",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    if not args.shapefile.exists():
        logger.error("Shapefile not found: %s", args.shapefile)
        sys.exit(1)

    if args.input:
        if not args.input.exists():
            logger.error("Input directory not found: %s", args.input)
            sys.exit(1)
        input_paths = [args.input]
    else:
        for f in args.files:
            if not f.exists():
                logger.error("File not found: %s", f)
                sys.exit(1)
        input_paths = list(args.files)

    try:
        results = process_nimrod_local(
            input_paths=input_paths,
            shapefile_path=args.shapefile,
            output_dir=args.output,
            max_workers=args.workers,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if results["errors"] > 0 and results["processed"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
