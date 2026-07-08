#!/usr/bin/env python3
"""
delineate_catchment.py

Run the full pysheds watershed-delineation pipeline for a single pour point
on a DEM, producing a GeoJSON catchment polygon and a summary printed to
stdout.

This script works around a NumPy 2.x (NEP-50) incompatibility in pysheds
0.4 where ``np.can_cast(nodata, dtype, casting='safe')`` raises ``TypeError``
for Python scalars when the array dtype is boolean or integer.  The fix is
to use explicit NumPy scalar types (``np.int64``, ``np.bool_``) as the
``nodata_out`` argument for every pysheds step that produces an integer or
boolean output raster.

Pipeline
--------
1.  Load DEM with rasterio (metadata) and pysheds (Grid / Raster)
2.  Fill pits           – remove single-cell sinks
3.  Fill depressions    – fill multi-cell sinks so every cell drains to edge
4.  Resolve flats       – add tiny gradient across flat areas
5.  Flow direction (D8) – ``nodata_out=np.int64(0)``
6.  Flow accumulation   – ``nodata_out=np.int64(0)``
7.  Snap pour point     – nearest cell where acc > threshold
8.  Delineate catchment – ``nodata_out=np.bool_(False)``
9.  Vectorise           – polygonise the boolean mask → Shapely Polygon
10. Save GeoJSON        – write output file

Usage
-----
    python -m scripts.delineate_catchment \\
        --lon -3.4950 --lat 50.8140 \\
        --dem data/devon_dem.tif \\
        --output catchment.geojson

    # Adjust the snap threshold (default 5000 upstream cells)
    python -m scripts.delineate_catchment \\
        --lon -3.4950 --lat 50.8140 \\
        --dem data/devon_dem.tif \\
        --threshold 10000 \\
        --output catchment.geojson
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import mapping, shape

from eoflow.log_utils import get_logger

# ---------------------------------------------------------------------------
# NumPy 2.x compatibility shim
# ---------------------------------------------------------------------------
# ``np.in1d`` was deprecated in NumPy 2.0 in favour of ``np.isin`` and has
# been removed entirely in later NumPy 2.x releases (e.g. 2.4).  pysheds 0.4
# still calls ``np.in1d`` internally (pgrid.py / sgrid.py), so we restore it
# as a thin alias before importing pysheds.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from pysheds.grid import Grid  # noqa: E402  (must come after the np.in1d shim)

logger = get_logger(__file__)

# ---------------------------------------------------------------------------
# D8 direction map  (N, NE, E, SE, S, SW, W, NW)
# ---------------------------------------------------------------------------
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def delineate(
    lon: float,
    lat: float,
    dem_path: Path,
    flow_acc_threshold: int = 5000,
    pit_fill_eps: float = 0.0001,
) -> dict:
    """Run the full delineation pipeline and return a result dict.

    Parameters
    ----------
    lon, lat:
        Pour-point coordinates in the same CRS as the DEM (EPSG:4326 for
        standard Devon DEMs — longitude, latitude).
    dem_path:
        Path to the GeoTIFF DEM.
    flow_acc_threshold:
        Minimum upstream-cell count for a cell to be considered part of the
        stream network when snapping the pour point.
    pit_fill_eps:
        Epsilon (m) added when resolving flat areas.

    Returns
    -------
    dict with keys:
        pour_point        – original (lon, lat)
        snapped_point     – snapped (lon, lat)
        snap_offset_m     – distance moved during snapping (metres, approx.)
        flow_acc_at_snap  – accumulation value at the snapped cell
        polygon           – Shapely Polygon
        area_km2          – approximate catchment area in km²
        bounds            – (minx, miny, maxx, maxy)
        elapsed_s         – wall-clock seconds for the whole run
    """
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    t0 = time.perf_counter()

    # ── 1. Load DEM metadata via rasterio ────────────────────────────────
    with rasterio.open(dem_path) as src:
        dem_bounds = src.bounds
        dem_transform = src.transform
        dem_crs = src.crs
        dem_nodata = src.nodata

    logger.info("DEM  : %s", dem_path.name)
    logger.info("CRS  : %s", dem_crs)
    logger.info("Bounds: %s", dem_bounds)
    logger.info("Pour point: lon=%.6f  lat=%.6f", lon, lat)

    # Validate the pour point lies inside the DEM extent
    if not (
        dem_bounds.left <= lon <= dem_bounds.right and dem_bounds.bottom <= lat <= dem_bounds.top
    ):
        raise ValueError(
            f"Pour point ({lon}, {lat}) is outside DEM extent "
            f"({dem_bounds.left:.4f}–{dem_bounds.right:.4f}, "
            f"{dem_bounds.bottom:.4f}–{dem_bounds.top:.4f})"
        )

    # ── 2. Load with pysheds ──────────────────────────────────────────────
    logger.info("Loading DEM into pysheds …")
    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))

    # ── 3. Fill pits ──────────────────────────────────────────────────────
    logger.info("Step 1/6 — fill pits …")
    pit_filled = grid.fill_pits(dem)

    # ── 4. Fill depressions ───────────────────────────────────────────────
    logger.info("Step 2/6 — fill depressions …")
    flooded = grid.fill_depressions(pit_filled)

    # ── 5. Resolve flats ──────────────────────────────────────────────────
    logger.info("Step 3/6 — resolve flats (eps=%g) …", pit_fill_eps)
    inflated = grid.resolve_flats(flooded, eps=pit_fill_eps)

    # After fill_pits the DEM is upcast to float64.  pysheds propagates this
    # float nodata into every downstream Raster, but integer-typed outputs
    # (flowdir, accumulation) and boolean outputs (catchment) cannot store a
    # float nodata value.  Under NumPy 2.x (NEP 50) np.can_cast raises
    # TypeError for Python scalars, so we must pass explicit NumPy scalars.
    _int_kw = {"nodata_out": np.int64(0)}  # for flowdir & accumulation
    _bool_kw = {"nodata_out": np.bool_(False)}  # for catchment

    # ── 6. Flow direction (D8) ────────────────────────────────────────────
    logger.info("Step 4/6 — D8 flow direction …")
    fdir = grid.flowdir(inflated, dirmap=DIRMAP, routing="d8", **_int_kw)

    # ── 7. Flow accumulation ──────────────────────────────────────────────
    logger.info("Step 5/6 — flow accumulation …")
    acc = grid.accumulation(fdir, dirmap=DIRMAP, routing="d8", **_int_kw)

    acc_np = np.array(acc, dtype=float)
    acc_np[acc_np == 0] = np.nan
    logger.info("  Max accumulation: %,.0f cells", float(np.nanmax(acc_np)))

    # ── 8. Snap pour point to stream network ─────────────────────────────
    logger.info("Step 6/6 — snap pour point (threshold=%d cells) …", flow_acc_threshold)

    # acc > threshold produces a pysheds Raster with int64 nodata; pysheds'
    # snap_to_mask internally re-views this as bool against the grid's
    # viewfinder.  This is safe now that sview.py wraps nodata in np.array().
    x_snap, y_snap = grid.snap_to_mask(
        acc > flow_acc_threshold,
        (lon, lat),
        return_dist=False,
    )

    # Snap offset in approximate metres
    lat_rad = math.radians((lat + y_snap) / 2)
    snap_offset_m = math.sqrt(
        ((x_snap - lon) * 111_320 * math.cos(lat_rad)) ** 2 + ((y_snap - lat) * 111_132) ** 2
    )
    logger.info("  Original  : (%.6f, %.6f)", lon, lat)
    logger.info("  Snapped to: (%.6f, %.6f)  offset≈%.0f m", x_snap, y_snap, snap_offset_m)

    # Accumulation value at the snapped cell
    col_snap, row_snap = grid.nearest_cell(x_snap, y_snap)
    flow_acc_at_snap = float(acc_np[int(row_snap), int(col_snap)])
    logger.info("  Flow acc at snap: %,.0f cells", flow_acc_at_snap)

    # ── 9. Delineate catchment ────────────────────────────────────────────
    logger.info("Delineating catchment …")
    catch = grid.catchment(
        x=x_snap,
        y=y_snap,
        fdir=fdir,
        dirmap=DIRMAP,
        routing="d8",
        xytype="coordinate",
        **_bool_kw,
    )

    catch_np = np.array(catch, dtype=bool)
    logger.info("  Catchment cells: %d", int(catch_np.sum()))

    # ── 10. Vectorise ─────────────────────────────────────────────────────
    logger.info("Vectorising catchment mask …")
    polygons = [shape(geom) for geom, val in grid.polygonize(catch.astype(np.int32)) if val == 1]

    if not polygons:
        raise RuntimeError(
            "No catchment polygon produced — the pour point may be outside "
            "the valid DEM area or the threshold may be too high."
        )

    polygon = max(polygons, key=lambda p: p.area)

    # Approximate area in km²
    centroid_lat = polygon.centroid.y
    area_km2 = polygon.area * 111_132.0 * 111_320.0 * math.cos(math.radians(centroid_lat)) / 1e6

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done in %.1f s — area≈%.1f km²  bounds=%s",
        elapsed,
        area_km2,
        polygon.bounds,
    )

    return {
        "pour_point": (lon, lat),
        "snapped_point": (float(x_snap), float(y_snap)),
        "snap_offset_m": snap_offset_m,
        "flow_acc_at_snap": flow_acc_at_snap,
        "polygon": polygon,
        "area_km2": area_km2,
        "bounds": polygon.bounds,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# GeoJSON output
# ---------------------------------------------------------------------------


def write_geojson(result: dict, output_path: Path) -> None:
    """Write the catchment polygon to a GeoJSON file."""
    feature = {
        "type": "Feature",
        "geometry": mapping(result["polygon"]),
        "properties": {
            "pour_lon": result["pour_point"][0],
            "pour_lat": result["pour_point"][1],
            "snapped_lon": result["snapped_point"][0],
            "snapped_lat": result["snapped_point"][1],
            "snap_offset_m": round(result["snap_offset_m"], 1),
            "flow_acc_at_snap": result["flow_acc_at_snap"],
            "area_km2": round(result["area_km2"], 2),
        },
    }
    fc = {"type": "FeatureCollection", "features": [feature]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(fc, fh, indent=2)
    logger.info("GeoJSON written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delineate a catchment from a pour point using pysheds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--lon", type=float, required=True, help="Pour-point longitude (EPSG:4326).")
    p.add_argument("--lat", type=float, required=True, help="Pour-point latitude (EPSG:4326).")
    p.add_argument(
        "--dem", type=Path, required=True, metavar="PATH", help="Path to the GeoTIFF DEM."
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=5000,
        metavar="CELLS",
        help="Flow-accumulation threshold for stream snapping.",
    )
    p.add_argument(
        "--eps", type=float, default=0.0001, metavar="EPS", help="Epsilon for flat resolution."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("catchment.geojson"),
        metavar="PATH",
        help="Output GeoJSON path.",
    )
    p.add_argument(
        "--no-save", action="store_true", help="Skip writing the GeoJSON file (print summary only)."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        result = delineate(
            lon=args.lon,
            lat=args.lat,
            dem_path=args.dem,
            flow_acc_threshold=args.threshold,
            pit_fill_eps=args.eps,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        sys.exit(1)

    # ── Print summary ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Catchment delineation summary")
    print("=" * 60)
    print(f"  Pour point   : ({result['pour_point'][0]:.6f}, {result['pour_point'][1]:.6f})")
    print(f"  Snapped to   : ({result['snapped_point'][0]:.6f}, {result['snapped_point'][1]:.6f})")
    print(f"  Snap offset  : {result['snap_offset_m']:.0f} m")
    print(f"  Flow acc     : {result['flow_acc_at_snap']:,.0f} upstream cells")
    print(f"  Area         : {result['area_km2']:.2f} km²")
    print(f"  Bounds       : {result['bounds']}")
    print(f"  Elapsed      : {result['elapsed_s']:.1f} s")
    print("=" * 60)
    print()

    if not args.no_save:
        try:
            write_geojson(result, args.output)
            print(f"GeoJSON saved to: {args.output}")
        except Exception as exc:
            logger.error("Failed to write GeoJSON: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
