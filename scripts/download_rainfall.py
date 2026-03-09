"""
Download Met Office UKV 2 km rainfall-rate NetCDF files from the AWS Open
Data S3 bucket.

This script is a thin CLI wrapper around :func:`eoflow.rainfall.download_rainfall`.
No AWS account or credentials are required.

Bucket
------
``s3://met-office-atmospheric-model-data/uk-deterministic-2km/``

Usage
-----
::

    python -m scripts.download_rainfall \\
        --start 2024-03-01T00:00 \\
        --end   2024-03-01T06:00 \\
        --output ./rainfall_data

    # Pin to the midnight (00Z) model run only
    python -m scripts.download_rainfall \\
        --start 2024-03-01T00:00 \\
        --end   2024-03-01T06:00 \\
        --run-hour 0 \\
        --output ./rainfall_data

    # Dry-run (list files without downloading)
    python -m scripts.download_rainfall \\
        --start 2024-03-01T00:00 \\
        --end   2024-03-01T06:00 \\
        --dry-run

File-naming convention on S3
-----------------------------
``uk-deterministic-2km/<run_time>/<valid_time>-PT<lead>H<mm>M-rainfall_rate.nc``

For example::

    20240301T0000Z/20240301T0000Z-PT0000H00M-rainfall_rate.nc
    20240301T0000Z/20240301T0015Z-PT0000H15M-rainfall_rate.nc
    20240301T0000Z/20240301T0100Z-PT0001H00M-rainfall_rate.nc

The script maps each 15-minute valid-time step within ``[start, end]`` to the
corresponding model-run directory on S3 and downloads the matching
``rainfall_rate.nc`` file.  Where multiple model runs cover the same valid
time the most recent run is preferred.

Common model-run hours
----------------------
* Nowcast runs: 01, 02, 04, 05, 07, 08, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23
* Short runs  : 00, 06, 09, 12, 18, 21
* Medium runs : 03, 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eoflow.log_utils import get_logger
from eoflow.rainfall import download_rainfall, parse_datetime

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download Met Office UKV rainfall_rate NetCDF files from the "
            "AWS Open Data S3 bucket (no credentials required)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM-DDTHH:MM",
        help="Start of the valid-time range (UTC, inclusive).",
    )
    p.add_argument(
        "--end",
        required=True,
        metavar="YYYY-MM-DDTHH:MM",
        help="End of the valid-time range (UTC, inclusive).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("./rainfall_data"),
        metavar="DIR",
        help="Local output directory (default: %(default)s).",
    )
    p.add_argument(
        "--run-hour",
        type=int,
        default=None,
        metavar="HOUR",
        help=(
            "If set, only use files from the model run starting at this UTC "
            "hour (0–23).  Useful to pin to a specific run type, e.g. "
            "--run-hour 0 for the midnight short run."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel download threads (default: %(default)s).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without downloading them.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validate datetimes early so the user gets a clear error message.
    try:
        start_dt = parse_datetime(args.start)
    except ValueError as exc:
        logger.error("Invalid --start value: %s", exc)
        sys.exit(1)

    try:
        end_dt = parse_datetime(args.end)
    except ValueError as exc:
        logger.error("Invalid --end value: %s", exc)
        sys.exit(1)

    if end_dt < start_dt:
        logger.error("--end must be >= --start")
        sys.exit(1)

    if args.run_hour is not None and not (0 <= args.run_hour <= 23):
        logger.error("--run-hour must be between 0 and 23, got %d", args.run_hour)
        sys.exit(1)

    try:
        counts = download_rainfall(
            start=start_dt,
            end=end_dt,
            output_dir=args.output,
            run_hour=args.run_hour,
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

    # Surface a non-zero exit code if every attempted file errored.
    if counts["error"] > 0 and counts["downloaded"] == 0 and counts["skipped"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
