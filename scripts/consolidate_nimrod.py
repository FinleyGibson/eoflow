"""
Consolidate per-timestep NIMROD NetCDF files into a single merged NetCDF.

This script is a thin CLI wrapper around
:func:`eoflow.rainfall.consolidate_nimrod`.

Each input file is a single-timestep NetCDF produced by
``process_nimrod_local.py``.  This script merges them along the time
dimension, producing one file with shape
``(time, projection_y_coordinate, projection_x_coordinate)``.

Files are opened in parallel using ``dask.delayed`` (``parallel=True``) and
a ``preprocess`` step promotes the scalar ``time`` coordinate written by Iris
into a proper dimension before concatenation, making this significantly faster
than a naive ``open_mfdataset`` call.

Usage
-----
::

    # Merge all processed files under a directory tree
    python -m scripts.consolidate_nimrod \\
        --input  data/nimrod_processed/2026 \\
        --output data/nimrod_2026.nc

    # Merge a specific date range only
    python -m scripts.consolidate_nimrod \\
        --input  data/nimrod_processed/2026 \\
        --output data/nimrod_jan2026.nc \\
        --start  2026-01-01 \\
        --end    2026-01-31

    # Dry-run: show which files would be merged without writing anything
    python -m scripts.consolidate_nimrod \\
        --input  data/nimrod_processed/2026 \\
        --output data/nimrod_2026.nc \\
        --dry-run

    # Increase compression (default 4; range 0-9)
    python -m scripts.consolidate_nimrod \\
        --input  data/nimrod_processed/2026 \\
        --output data/nimrod_2026.nc \\
        --compression 6

Input layout
------------
The script searches *input* recursively for files matching ``*.nc``, as written
by ``process_nimrod_local.py``::

    <input>/
        <YYYY>/
            <YYYYMMDD>/
                <YYYYMMDD_HHMMSS>.nc
                ...

Files are sorted alphabetically before merging; because the stem format is
``YYYYMMDD_HHMMSS``, alphabetical order equals chronological order.

Output
------
A single NetCDF4 file with dimensions
``(time, projection_y_coordinate, projection_x_coordinate)``
and lossless zlib compression applied to all data variables.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from eoflow.log_utils import get_logger
from eoflow.rainfall import consolidate_nimrod

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_date(value: str, flag: str) -> datetime:
    """Parse YYYY-MM-DD or YYYYMMDD into a datetime, or exit with a clear error."""
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    logger.error("--%s: cannot parse date %r — expected YYYY-MM-DD or YYYYMMDD", flag, value)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Consolidate per-timestep NIMROD NetCDF files into a single merged NetCDF4 file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="DIR",
        help="Root directory containing per-timestep .nc files (searched recursively).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="FILE",
        help="Destination .nc file to write.",
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Include only timesteps on or after this date (optional).",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Include only timesteps on or before this date (optional).",
    )
    p.add_argument(
        "--compression",
        type=int,
        default=4,
        metavar="0-9",
        help="zlib compression level for the output file (default: %(default)s; 0 = none).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which files would be merged without writing output.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input.exists():
        logger.error("Input directory does not exist: %s", args.input)
        sys.exit(1)

    if not 0 <= args.compression <= 9:
        logger.error("--compression must be between 0 and 9, got %d", args.compression)
        sys.exit(1)

    start = parse_date(args.start, "start") if args.start else None
    end = parse_date(args.end, "end") if args.end else None

    if start and end and start > end:
        logger.error("--start (%s) must not be after --end (%s)", start.date(), end.date())
        sys.exit(1)

    try:
        results = consolidate_nimrod(
            input_dir=args.input,
            output_path=args.output,
            start=start,
            end=end,
            compression=args.compression,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if results["merged"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
