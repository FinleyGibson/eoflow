"""
Consolidate per-timestep NIMROD NetCDF files into a single merged NetCDF.

Each input file is a single-timestep NetCDF produced by ``process_nimrod_local.py``,
with a proper CF-compliant ``time`` coordinate embedded in the file.  This script
merges them along the time dimension, producing one file with shape
``(time, projection_y_coordinate, projection_x_coordinate)``.

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

    # Increase compression (default 4; range 0–9)
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
The ``time`` coordinate is a proper CF datetime axis derived from the
embedded coordinates in each source file — not the filenames.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import xarray as xr
from tqdm.dask import TqdmCallback

from eoflow.log_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_nc_files(
    input_dir: Path,
    start: datetime | None,
    end: datetime | None,
) -> tuple[list[Path], int]:
    """Return sorted NetCDF paths, optionally filtered to a date range.

    Files are assumed to have stems of the form ``YYYYMMDD_HHMMSS``; any
    file whose stem cannot be parsed is always included.

    Parameters
    ----------
    input_dir:
        Root directory to search recursively.
    start:
        Inclusive lower bound (local time).  ``None`` means no lower bound.
    end:
        Inclusive upper bound (local time).  ``None`` means no upper bound.

    Returns
    -------
    tuple[list[Path], int]
        ``(matched_files, n_skipped_by_filter)``
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
            # Filename doesn't match expected pattern — include it anyway.
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


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def consolidate_nimrod(
    input_dir: Path,
    output_path: Path,
    start: datetime | None = None,
    end: datetime | None = None,
    compression: int = 4,
    dry_run: bool = False,
) -> dict[str, int]:
    """Merge per-timestep NIMROD NetCDF files into a single NetCDF4 file.

    Parameters
    ----------
    input_dir:
        Root directory containing per-timestep ``.nc`` files (searched recursively).
    output_path:
        Destination ``.nc`` file to write.
    start:
        Optional inclusive start datetime for date-range filtering.
    end:
        Optional inclusive end datetime for date-range filtering.
    compression:
        zlib compression level 0–9 applied to all data variables (0 = none).
    dry_run:
        If ``True``, print what would be done without writing anything.

    Returns
    -------
    dict
        ``{"merged": int, "skipped": int}``
    """
    logger.info("Searching for NetCDF files in: %s", input_dir)
    nc_files, n_skipped = find_nc_files(input_dir, start, end)

    if not nc_files:
        logger.error("No NetCDF files found in: %s", input_dir)
        return {"merged": 0, "skipped": n_skipped}

    logger.info("Found %d file(s) to merge (%d filtered out)", len(nc_files), n_skipped)

    # -----------------------------------------------------------------------
    # Summary print
    # -----------------------------------------------------------------------
    date_range_line = ""
    if start or end:
        lo = start.date() if start else "any"
        hi = end.date() if end else "any"
        date_range_line = f"  Date range : {lo} -> {hi}\n"

    print(
        f"\n"
        f"  Input      : {input_dir}\n"
        f"  Output     : {output_path}\n"
        f"  Files      : {len(nc_files):,}  ({n_skipped:,} filtered out)\n"
        f"{date_range_line}"
        f"  Compression: zlib level {compression}\n"
    )

    if dry_run:
        logger.info("[dry-run] Would merge %d files -> %s", len(nc_files), output_path)
        for f in nc_files:
            try:
                rel = f.relative_to(input_dir)
            except ValueError:
                rel = f
            print(f"  {rel}")
        return {"merged": len(nc_files), "skipped": n_skipped}

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------
    logger.info("Opening %d file(s) with xarray...", len(nc_files))
    try:
        ds = xr.open_mfdataset(
            [str(f) for f in nc_files],
            # iris saves single-timestep cubes with a scalar (0-d) time
            # coordinate, so combine="by_coords" has nothing to sort by.
            # combine="nested" + concat_dim="time" instead stacks files in
            # the order given (already alphabetical = chronological) and
            # promotes the scalar time into a proper dimension coordinate.
            combine="nested",
            concat_dim="time",
            # Keep data lazy so large datasets don't blow memory.
            # Chunk by day's worth of 5-minute scans (288 per day).
            chunks={"time": 288},
            # Tolerate minor attribute differences between files
            # (e.g. iris sometimes writes slightly different history strings).
            compat="override",
            coords="minimal",
        )
    except Exception as exc:
        logger.error("Failed to open/merge files: %s", exc, exc_info=True)
        raise

    time_start = str(ds.time.values[0])[:19]
    time_end = str(ds.time.values[-1])[:19]

    logger.info("Dataset summary:")
    logger.info("  Dimensions : %s", dict(ds.sizes))
    logger.info("  Variables  : %s", list(ds.data_vars))
    logger.info("  Time range : %s -> %s", time_start, time_end)
    logger.info("  Timesteps  : %d", ds.sizes.get("time", "?"))

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoding = {var: {"zlib": compression > 0, "complevel": compression} for var in ds.data_vars}

    logger.info("Writing merged dataset to: %s", output_path)

    with TqdmCallback(desc="Writing", unit="chunk"):
        ds.to_netcdf(
            output_path,
            format="NETCDF4",
            encoding=encoding,
        )

    size_mb = output_path.stat().st_size / (1024**2)
    logger.info("Done. Output: %s (%.1f MB)", output_path, size_mb)

    return {"merged": len(nc_files), "skipped": n_skipped}


# ---------------------------------------------------------------------------
# CLI
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
