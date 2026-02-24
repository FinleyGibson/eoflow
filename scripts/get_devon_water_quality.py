"""
Fetch water quality data from the Environment Agency API for Devon only.

This script follows the same month-by-month checkpointed approach as
:mod:`scripts.get_ea_water_quality`, but additionally filters each month's
results to the Devon county boundary using
:meth:`eoflow.ea.EAWaterQualityAPI.filter_by_polygon`.

Because precanned EA area codes (e.g. ``environment_agency,SWX``) often
return 400 errors, data is fetched without a geographic filter and then
clipped to the Devon polygon extracted from a local shapefile.

Usage
-----
    python -m scripts.get_devon_water_quality \
        --determinand 0076 \
        --start-date 2023-01-01 \
        --end-date 2023-12-31 \
        --out data/devon_temperature_2023.csv

    # Optional flags
        --shapefile data/devon_county       # path to Devon shapefile dir
        --delay 0.5                         # seconds between API requests
        --timeout 60                        # per-request timeout
        --checkpoint-dir .checkpoints       # where to keep state files
        --verbose / --quiet                 # control console output

Checkpoint / crash-recovery
---------------------------
Each run derives a deterministic *run key* from (determinand, start_date,
end_date, "devon").  Completed months are recorded in a JSON state file
under ``--checkpoint-dir``.  If the script is re-started with the same
parameters, it skips already-fetched months and appends only the missing
ones to the output CSV.

Common determinand codes
------------------------
* ``0076`` – Temperature of Water
* ``0077`` – Conductivity at 25 °C
* ``0180`` – Orthophosphate, reactive as P
* ``6396`` – Turbidity (NTU)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import geopandas as gpd
import pandas as pd

from eoflow.ea import EAWaterQualityAPI
from eoflow.log_utils import get_logger
from eoflow.utils import load_shapefile

logger = get_logger(__name__)

# Default shapefile path (relative to the repository root)
_DEFAULT_SHAPEFILE = Path(__file__).resolve().parent.parent / "data" / "devon_county"

# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------


def extract_polygon_from_geodataframe(
    gdf: gpd.GeoDataFrame,
) -> List[Tuple[float, float]]:
    """Extract polygon coordinates from a GeoDataFrame.

    If the geometry is a MultiPolygon the largest polygon (by area) is used.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing at least one feature.

    Returns
    -------
    list of (lon, lat) tuples
    """
    geom = gdf.geometry.iloc[0]

    if geom.geom_type == "MultiPolygon":
        logger.info("Geometry is MultiPolygon – using largest polygon")
        largest_poly = max(geom.geoms, key=lambda p: p.area)
        coords = list(largest_poly.exterior.coords)
    elif geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
    else:
        raise ValueError(f"Unexpected geometry type: {geom.geom_type}")

    return coords


def load_devon_polygon(shapefile_path: Path) -> List[Tuple[float, float]]:
    """Load the Devon county shapefile and return its polygon in WGS84.

    Parameters
    ----------
    shapefile_path : Path
        Path to the shapefile (directory or ``.shp`` file).

    Returns
    -------
    list of (lon, lat) tuples
    """
    gdf = load_shapefile(shapefile_path)
    logger.info("Loaded shapefile with %d feature(s)", len(gdf))

    # Reproject to WGS84 if necessary
    if gdf.crs is not None and gdf.crs != "EPSG:4326":
        logger.info("Reprojecting from %s to EPSG:4326", gdf.crs)
        gdf = gdf.to_crs("EPSG:4326")

    polygon_coords = extract_polygon_from_geodataframe(gdf)
    logger.info("Devon polygon extracted: %d vertices", len(polygon_coords))
    return polygon_coords


# ---------------------------------------------------------------------------
# State / checkpoint helpers
# ---------------------------------------------------------------------------

_STATE_VERSION = 1


def _run_key(
    determinand: str,
    start_date: str,
    end_date: str,
) -> str:
    """Return a short, filesystem-safe hash identifying a unique run."""
    raw = f"{determinand}|devon|{start_date}|{end_date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _state_path(checkpoint_dir: Path, key: str) -> Path:
    return checkpoint_dir / f"devon_wq_{key}.state.json"


def _load_state(path: Path) -> Dict[str, Any]:
    """Load checkpoint state from disk, or return a fresh state dict."""
    if path.exists():
        try:
            with open(path) as fh:
                state = json.load(fh)
            logger.info("Loaded checkpoint state from %s", path)
            return state
        except Exception:
            logger.warning("Could not read state file %s – starting fresh", path)
    return {"version": _STATE_VERSION, "completed_months": []}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(state, fh, indent=2)
    logger.debug("State saved to %s", path)


def _completed_set(state: Dict[str, Any]) -> Set[str]:
    """Return a set of 'YYYY-MM-DD/YYYY-MM-DD' strings for completed months."""
    return set(state.get("completed_months", []))


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    """Append rows to a CSV file, writing headers only if the file is new."""
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", index=False, header=write_header)


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------


def fetch_devon_water_quality(
    determinand: str,
    start_date: str,
    end_date: str,
    output_path: Path,
    *,
    shapefile_path: Path = _DEFAULT_SHAPEFILE,
    delay: float = 0.5,
    timeout: int = 60,
    checkpoint_dir: Path = Path(".checkpoints"),
    verbose: bool = True,
) -> pd.DataFrame:
    """Download EA water quality data for Devon month-by-month with checkpointing.

    Data is fetched from the EA API without a geographic filter, then
    filtered to the Devon county boundary using
    :meth:`~eoflow.ea.EAWaterQualityAPI.filter_by_polygon`.

    Parameters
    ----------
    determinand : str
        EA determinand code (e.g. ``"0076"``).
    start_date, end_date : str
        Date range in ``YYYY-MM-DD`` format.
    output_path : Path
        Destination CSV file.  Data is appended after each month.
    shapefile_path : Path
        Path to the Devon county shapefile (directory or ``.shp`` file).
    delay : float
        Seconds to wait between API requests.
    timeout : int
        Per-request timeout in seconds.
    checkpoint_dir : Path
        Directory for JSON state files.
    verbose : bool
        Whether to print progress to the console.

    Returns
    -------
    pandas.DataFrame
        The combined data (all months, including previously checkpointed ones).
    """
    # ------------------------------------------------------------------
    # Load the Devon polygon
    # ------------------------------------------------------------------
    if verbose:
        print(f"\nLoading Devon boundary from {shapefile_path} …")

    polygon_coords = load_devon_polygon(shapefile_path)

    if verbose:
        print(f"  ✓ Devon polygon loaded ({len(polygon_coords)} vertices)")

    # ------------------------------------------------------------------
    # Set up checkpointing
    # ------------------------------------------------------------------
    key = _run_key(determinand, start_date, end_date)
    state_file = _state_path(checkpoint_dir, key)
    state = _load_state(state_file)
    completed = _completed_set(state)

    api = EAWaterQualityAPI(delay=delay, timeout=timeout)

    # Generate the full list of month windows.
    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    month_ranges: List[Tuple[datetime, datetime]] = api._generate_month_ranges(dt_start, dt_end)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(month_ranges)
    n_skipped = 0
    n_fetched = 0
    n_records = 0

    if verbose:
        print("\nDevon Water Quality data download")
        print(f"  Determinand : {determinand}")
        print(f"  Date range  : {start_date} → {end_date}")
        print("  Region      : Devon (polygon filter)")
        print(f"  Output      : {output_path}")
        print(f"  Months      : {n_total}")
        already = len(completed)
        if already:
            print(f"  Resuming    : {already} month(s) already fetched")
        print()

    # ------------------------------------------------------------------
    # Month-by-month download and filter
    # ------------------------------------------------------------------
    for idx, (m_start, m_end) in enumerate(month_ranges, start=1):
        month_key = f"{m_start:%Y-%m-%d}/{m_end:%Y-%m-%d}"

        if month_key in completed:
            n_skipped += 1
            if verbose:
                logger.info("[%d/%d] %s – already fetched, skipping", idx, n_total, month_key)
            continue

        if verbose:
            logger.info("[%d/%d] Fetching %s …", idx, n_total, month_key)

        # Fetch without area filter – we rely on polygon filtering instead.
        try:
            df = api._make_request(
                determinand=determinand,
                date_from=f"{m_start:%Y-%m-%d}",
                date_to=f"{m_end:%Y-%m-%d}",
                precanned_area=None,
            )
        except Exception as exc:
            logger.error("  ✗ Request failed for %s: %s", month_key, exc)
            # Don't mark as completed – it will be retried on the next run.
            continue

        rows = 0
        if df is not None and not df.empty:
            # Drop rows with missing results (consistent with get_data)
            df = df.dropna(subset=["result"])

            if not df.empty:
                # Filter to Devon polygon
                try:
                    df = api.filter_by_polygon(df=df, polygon=polygon_coords)
                except Exception as exc:
                    logger.error("  ✗ Polygon filter failed for %s: %s", month_key, exc)
                    continue

                rows = len(df)
                if rows:
                    _append_csv(df, output_path)

        n_fetched += 1
        n_records += rows

        # Mark this month as completed and persist state.
        state["completed_months"].append(month_key)
        completed.add(month_key)
        _save_state(state_file, state)

        if verbose:
            logger.info("  ✓ %d records in Devon", rows)

        # Polite delay between requests (skip after last month).
        if idx < n_total:
            time.sleep(delay)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if verbose:
        print()
        print("=" * 60)
        print("Download complete")
        print(f"  Months fetched this run : {n_fetched}")
        print(f"  Months skipped (cached) : {n_skipped}")
        print(f"  Records written (Devon) : {n_records}")
        if output_path.exists():
            print(f"  Output file             : {output_path}")
        print("=" * 60)
        print()

    # Return the full combined dataset from the output CSV.
    if output_path.exists() and output_path.stat().st_size > 0:
        combined = pd.read_csv(output_path)
        logger.info("Total records in output file: %d", len(combined))
        return combined

    logger.warning("No data collected for Devon.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download water quality data from the Environment Agency API "
        "filtered to the Devon county boundary, with automatic "
        "checkpointing and crash recovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--determinand",
        required=True,
        help=(
            "EA determinand code.  Common codes: "
            "0076 (Temperature), 0077 (Conductivity), "
            "0180 (Orthophosphate), 6396 (Turbidity)."
        ),
    )
    p.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    p.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("devon_water_quality.csv"),
        help="Output CSV path.",
    )
    p.add_argument(
        "--shapefile",
        type=Path,
        default=_DEFAULT_SHAPEFILE,
        help="Path to the Devon county shapefile (directory or .shp file).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API requests.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(".checkpoints"),
        help="Directory for checkpoint state files.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress messages.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress messages.",
    )
    args = p.parse_args(argv)
    if args.quiet:
        args.verbose = False
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validate dates
    for label, val in [("start-date", args.start_date), ("end-date", args.end_date)]:
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid %s: %s  (expected YYYY-MM-DD)", label, val)
            sys.exit(1)

    try:
        fetch_devon_water_quality(
            determinand=args.determinand,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.out,
            shapefile_path=args.shapefile,
            delay=args.delay,
            timeout=args.timeout,
            checkpoint_dir=args.checkpoint_dir,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted – progress has been checkpointed.")
        logger.info("Script interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\n✗ ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
