"""
Dataset builder: iterate over water quality samples from a CSV, delineate
catchments with pysheds, and produce a GeoDataFrame with catchment polygons.

Features
--------
* Caches catchment polygons per unique (lat, long) pair so duplicate
  locations are not re-computed.
* Periodically checkpoints the full result to a GeoPackage on disk.
* On restart, loads the checkpoint and skips rows that already have a
  catchment, so no work is repeated after a crash.

Usage
-----
    python -m scripts.dataset_builder \
        --csv  notebooks/citizen_science_data/cs_df_devon \
        --dem  /home/finley/Work/RDS/projects/enforce/data/rasters/elevation_raw.tif \
        --out  outputs/devon_catchments.gpkg

    # Optional flags
        --checkpoint-every  10        # save after every N new delineations
        --flow-acc-threshold 1000     # pysheds snap threshold
        --lat-col   lat               # column name for latitude
        --lon-col   long              # column name for longitude
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

from eoflow.catchment import delineate_catchment
from eoflow.log_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CHECKPOINT_LAYER = "catchments"
_CATCHMENT_WKT_COL = "__catchment_wkt"  # used only inside the checkpoint file


def _save_checkpoint(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Persist current state to a GeoPackage file.

    The GeoDataFrame's ``geometry`` column holds the catchment polygon (or
    None for rows not yet processed / failed).  We keep the original sample
    point as a WKT string so we don't lose it on reload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Work on a copy so we don't mutate the caller's frame.
    out = gdf.copy()

    # Ensure the geometry column is the catchment column.
    # Store the catchment WKT explicitly so we can detect "already done" rows
    # even when the geometry is None / empty.
    out[_CATCHMENT_WKT_COL] = out["catchment"].apply(
        lambda g: g.wkt if isinstance(g, Polygon) else ""
    )

    # GeoPackage requires a geometry column – use the catchment.
    # Rows with None geometry need a placeholder; gpd handles None natively.
    out = out.set_geometry("catchment")

    out.to_file(str(path), layer=CHECKPOINT_LAYER, driver="GPKG")
    logger.info("Checkpoint saved to %s  (%d rows)", path, len(out))


def _load_checkpoint(path: Path) -> Optional[gpd.GeoDataFrame]:
    """Load a previous checkpoint if it exists, else return None."""
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(str(path), layer=CHECKPOINT_LAYER)
        logger.info("Loaded checkpoint from %s  (%d rows)", path, len(gdf))
        # Rename the geometry column back to 'catchment'
        if gdf.geometry.name != "catchment":
            gdf = gdf.rename_geometry("catchment")
        # Drop the helper WKT column if present
        if _CATCHMENT_WKT_COL in gdf.columns:
            gdf = gdf.drop(columns=[_CATCHMENT_WKT_COL])
        return gdf
    except Exception:
        logger.warning("Could not load checkpoint – starting fresh.\n%s", traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _build_location_cache(
    gdf: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
) -> Dict[Tuple[float, float], Polygon]:
    """Build a dict mapping (lat, lon) → Polygon from already-processed rows."""
    cache: Dict[Tuple[float, float], Polygon] = {}
    for _, row in gdf.iterrows():
        geom = row.get("catchment")
        if isinstance(geom, Polygon):
            key = (float(row[lat_col]), float(row[lon_col]))
            cache[key] = geom
    return cache


def build_dataset(
    csv_path: Path,
    dem_path: Path,
    output_path: Path,
    *,
    lat_col: str = "lat",
    lon_col: str = "long",
    checkpoint_every: int = 10,
    flow_acc_threshold: int = 1000,
) -> gpd.GeoDataFrame:
    """Main entry point: load CSV, delineate catchments, save GeoPackage.

    Parameters
    ----------
    csv_path : Path
        Path to the input CSV with water-quality samples.  Must contain
        columns *lat_col* and *lon_col*.
    dem_path : Path
        Path to a GeoTIFF DEM that covers (at least some of) the sample
        locations.  Points outside the DEM extent are logged and skipped.
    output_path : Path
        Where to write the output GeoPackage.  Also used as the
        checkpoint path (the file is overwritten on each save).
    lat_col, lon_col : str
        Column names for latitude and longitude.
    checkpoint_every : int
        Save a checkpoint after this many *new* catchment delineations.
    flow_acc_threshold : int
        ``flow_acc_threshold`` forwarded to
        :func:`eoflow.catchment.delineate_catchment`.

    Returns
    -------
    geopandas.GeoDataFrame
        The completed dataset with a ``catchment`` geometry column and
        additional ``delineation_status`` and ``delineation_error`` columns.
    """

    # ------------------------------------------------------------------
    # 1. Load the input CSV
    # ------------------------------------------------------------------
    logger.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    n_total = len(df)
    logger.info("Loaded %d rows with %d columns", n_total, len(df.columns))

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{lat_col}' and '{lon_col}' columns.  Found: {list(df.columns)}"
        )

    # Drop rows without coordinates
    valid_mask = df[lat_col].notna() & df[lon_col].notna()
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        logger.warning("Dropping %d rows with missing coordinates", n_dropped)
    df = df[valid_mask].copy()

    # ------------------------------------------------------------------
    # 2. Try to load a previous checkpoint
    # ------------------------------------------------------------------
    checkpoint = _load_checkpoint(output_path)

    if checkpoint is not None and len(checkpoint) == len(df):
        # Checkpoint row count matches – reuse it directly.
        gdf = checkpoint
        logger.info("Checkpoint matches input length – resuming")
    else:
        if checkpoint is not None:
            logger.warning(
                "Checkpoint length (%d) differs from CSV (%d).  "
                "Will rebuild, re-using cached catchment geometries where "
                "coordinates match.",
                len(checkpoint),
                len(df),
            )
        # Create a fresh GeoDataFrame from the CSV rows.
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries([None] * len(df)), crs="EPSG:4326")
        gdf = gdf.rename_geometry("catchment")
        gdf["delineation_status"] = ""
        gdf["delineation_error"] = ""

        # If we had a partial checkpoint with a different length, we can
        # still salvage cached polygons keyed by (lat, lon).
        if checkpoint is not None:
            old_cache = _build_location_cache(checkpoint, lat_col, lon_col)
            if old_cache:
                logger.info(
                    "Recovered %d cached catchment(s) from stale checkpoint",
                    len(old_cache),
                )
                for idx, row in gdf.iterrows():
                    key = (float(row[lat_col]), float(row[lon_col]))
                    if key in old_cache:
                        gdf.at[idx, "catchment"] = old_cache[key]
                        gdf.at[idx, "delineation_status"] = "ok"

    # Build a location cache from whatever we already have
    cache = _build_location_cache(gdf, lat_col, lon_col)
    logger.info("Location cache: %d unique locations already delineated", len(cache))

    # ------------------------------------------------------------------
    # 3. Iterate and delineate
    # ------------------------------------------------------------------
    dem_path = Path(dem_path)
    new_delineations = 0
    n_fail = 0

    unique_locations = gdf[[lat_col, lon_col]].drop_duplicates().values.tolist()
    locations_to_process = [(lat, lon) for lat, lon in unique_locations if (lat, lon) not in cache]
    logger.info(
        "%d unique locations total, %d still need delineation",
        len(unique_locations),
        len(locations_to_process),
    )

    for loc_idx, (lat, lon) in enumerate(locations_to_process, start=1):
        point = Point(lon, lat)  # shapely Point is (x=lon, y=lat)
        loc_key = (lat, lon)

        logger.info(
            "[%d/%d] Delineating catchment for (%.6f, %.6f) …",
            loc_idx,
            len(locations_to_process),
            lat,
            lon,
        )
        t0 = time.perf_counter()
        try:
            polygon = delineate_catchment(
                point=point,
                dem_path=dem_path,
                flow_acc_threshold=flow_acc_threshold,
            )
            elapsed = time.perf_counter() - t0
            cache[loc_key] = polygon
            new_delineations += 1
            status = "ok"
            error_msg = ""
            logger.info("  ✓ Delineated in %.1fs  (area=%.6f)", elapsed, polygon.area)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            status = "error"
            error_msg = str(exc)
            n_fail += 1
            logger.warning("  ✗ Failed in %.1fs: %s", elapsed, exc)

        # Apply result to every row sharing this (lat, lon)
        mask = (gdf[lat_col] == lat) & (gdf[lon_col] == lon)
        if status == "ok":
            gdf.loc[mask, "catchment"] = cache[loc_key]
        gdf.loc[mask, "delineation_status"] = status
        gdf.loc[mask, "delineation_error"] = error_msg

        # Periodic checkpoint
        if new_delineations > 0 and new_delineations % checkpoint_every == 0:
            logger.info("Checkpoint: %d new delineations so far …", new_delineations)
            _save_checkpoint(gdf, output_path)

    # ------------------------------------------------------------------
    # 4. Final save
    # ------------------------------------------------------------------
    n_ok = int((gdf["delineation_status"] == "ok").sum())
    n_err = int((gdf["delineation_status"] == "error").sum())
    n_empty = int((gdf["delineation_status"] == "").sum())

    logger.info("=" * 60)
    logger.info("Done.  %d rows total", len(gdf))
    logger.info("  OK:      %d rows  (%d unique locations)", n_ok, len(cache))
    logger.info("  Error:   %d rows", n_err)
    logger.info("  Skipped: %d rows (no coordinates / unprocessed)", n_empty)
    logger.info("  New delineations this run: %d", new_delineations)
    logger.info("=" * 60)

    _save_checkpoint(gdf, output_path)
    logger.info("Final output written to %s", output_path)

    return gdf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a GeoDataFrame of water-quality samples with "
        "delineated catchment polygons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the input CSV with water-quality samples.",
    )
    p.add_argument(
        "--dem",
        type=Path,
        required=True,
        help="Path to a GeoTIFF DEM covering the sample locations.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/devon_catchments.gpkg"),
        help="Output GeoPackage path (also used as checkpoint).",
    )
    p.add_argument(
        "--lat-col",
        default="lat",
        help="Name of the latitude column in the CSV.",
    )
    p.add_argument(
        "--lon-col",
        default="long",
        help="Name of the longitude column in the CSV.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint after this many new delineations.",
    )
    p.add_argument(
        "--flow-acc-threshold",
        type=int,
        default=1000,
        help="Flow-accumulation threshold for pour-point snapping.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.csv.exists():
        logger.error("CSV file not found: %s", args.csv)
        sys.exit(1)
    if not args.dem.exists():
        logger.error("DEM file not found: %s", args.dem)
        sys.exit(1)

    build_dataset(
        csv_path=args.csv,
        dem_path=args.dem,
        output_path=args.out,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        checkpoint_every=args.checkpoint_every,
        flow_acc_threshold=args.flow_acc_threshold,
    )


if __name__ == "__main__":
    main()
