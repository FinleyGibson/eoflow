"""
Convert downloaded Met Office UKV 2 km rainfall-rate NetCDF files into ESRI
ASCII raster (.asc) files on a British National Grid (EPSG:27700) grid.

This script is a thin CLI wrapper around :func:`eoflow.rainfall.convert_rainfall`.

Usage
-----
::

    python -m scripts.convert_rainfall \\
        --input  ./rainfall_data \\
        --output ./asc_output

    # Restrict to a specific spatial extent (BNG metres: x_min y_min x_max y_max)
    python -m scripts.convert_rainfall \\
        --input  ./rainfall_data \\
        --output ./asc_output \\
        --extent 285000 54000 295000 70000

    # Override resolution (default 1000 m) and worker count
    python -m scripts.convert_rainfall \\
        --input   ./rainfall_data \\
        --output  ./asc_output \\
        --res     2000 \\
        --workers 8

    # Dry-run (show what would be converted without doing it)
    python -m scripts.convert_rainfall \\
        --input  ./rainfall_data \\
        --output ./asc_output \\
        --dry-run

Input layout
------------
The script searches *input* recursively for files matching
``*rainfall_rate.nc``, as downloaded by ``scripts.download_rainfall``.

Output layout
-------------
``<output>/<run_YYYYMMDDHHMM>/<valid_YYYYMMDDHHMM>.asc``

This matches the naming convention used by ``trigger-aws.py`` so the ``.asc``
files can be dropped straight into the same downstream workflow.

Coordinate system
-----------------
All output rasters are projected to **EPSG:27700** (British National Grid).
The default extent covers the full UK domain::

    x_min=0  y_min=0  x_max=700000  y_max=1300000  (BNG metres)

Pass ``--extent`` to restrict the domain, e.g. to match the area used by
``trigger-aws.py``::

    --extent 285000 54000 295000 70000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eoflow.log_utils import get_logger
from eoflow.rainfall import UK_BNG_EXTENT, convert_rainfall

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert Met Office UKV rainfall_rate NetCDF files to "
            "British National Grid (EPSG:27700) ASCII rasters."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="DIR",
        help=(
            "Root directory containing downloaded *rainfall_rate.nc files. Searched recursively."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory to write .asc files into.",
    )
    p.add_argument(
        "--extent",
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        type=float,
        default=None,
        help=(
            "Spatial extent in BNG metres (EPSG:27700): x_min y_min x_max y_max. "
            f"Default: full UK {UK_BNG_EXTENT}. "
            "Example: --extent 285000 54000 295000 70000"
        ),
    )
    p.add_argument(
        "--res",
        type=int,
        default=1000,
        metavar="METRES",
        help="Output grid resolution in metres (default: %(default)s).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel conversion processes (default: %(default)s).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be converted without doing it.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validate input directory early for a clear error message.
    if not args.input.exists():
        logger.error("Input directory does not exist: %s", args.input)
        sys.exit(1)

    if args.res <= 0:
        logger.error("--res must be a positive integer, got %d", args.res)
        sys.exit(1)

    if args.workers < 1:
        logger.error("--workers must be >= 1, got %d", args.workers)
        sys.exit(1)

    # Unpack the flat --extent list into a typed tuple if provided.
    extent = tuple(args.extent) if args.extent is not None else None

    try:
        counts = convert_rainfall(
            input_dir=args.input,
            output_dir=args.output,
            extent=extent,
            res=args.res,
            workers=args.workers,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Exit with a non-zero code if every attempted conversion errored.
    if counts["error"] > 0 and counts["converted"] == 0 and counts["skipped"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
