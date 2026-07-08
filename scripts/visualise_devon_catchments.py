"""
Visualise water-quality sample points with their delineated catchment
polygons on an interactive map.

Reads the GeoPackage produced by ``delineate_catchments.py`` and (optionally)
a DEM GeoTIFF and boundary shapefile for the study area, then displays:

* An optional boundary outline (from a shapefile)
* A semi-transparent DEM elevation overlay (optional)
* A flow-accumulation overlay highlighting the drainage network (optional)
* Sample-point markers with informative popups, colour-coded by a value
  column (e.g. turbidity)
* Delineated catchment polygons for each sample point, hidden by default
  and revealed by clicking on the corresponding sample marker

The resulting HTML page is written to disk and opened automatically in
the default web browser.

Usage
-----
    python -m scripts.visualise_devon_catchments \
        --gpkg data/water_quality_dataset.gpkg \
        --dem data/dem.tif

    # With all options
    python -m scripts.visualise_devon_catchments \
        --gpkg data/water_quality_dataset.gpkg \
        --dem data/dem.tif \
        --shapefile data/study_area_boundary \
        --value-column result \
        --output catchments_map.html \
        --dem-opacity 0.5 \
        --max-pixels 2048 \
        --flow-acc-threshold 100 \
        --no-flow-acc
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import sys
import tempfile
import warnings
import webbrowser
from pathlib import Path
from typing import cast

import branca.colormap as cm
import folium
import folium.raster_layers  # noqa: F401
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from eoflow.catchment import compute_flow_accumulation
from eoflow.log_utils import get_logger
from eoflow.utils import load_shapefile

logger = get_logger(__file__)

# Mercator ↔ WGS 84 transformer (reused by DEM helpers)
_MERCATOR_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------------------
# DEM rendering helpers (adapted from visualise_dem.py)
# ---------------------------------------------------------------------------


def _make_terrain_cmap() -> mcolors.LinearSegmentedColormap:
    """Return a custom terrain colour-map (green → brown → white)."""
    colours = [
        (0.00, "#1a6e2e"),
        (0.15, "#5fa843"),
        (0.30, "#a1c34a"),
        (0.45, "#ddd07a"),
        (0.60, "#c4944a"),
        (0.75, "#9a7250"),
        (0.90, "#d4c8b8"),
        (1.00, "#f5f5f5"),
    ]
    positions = [c[0] for c in colours]
    hex_colours = [c[1] for c in colours]
    rgb = [mcolors.hex2color(h) for h in hex_colours]
    cdict: dict = {"red": [], "green": [], "blue": []}
    for pos, (r, g, b) in zip(positions, rgb):
        cdict["red"].append((pos, r, r))
        cdict["green"].append((pos, g, g))
        cdict["blue"].append((pos, b, b))
    return mcolors.LinearSegmentedColormap("terrain_custom", cdict, N=256)


def _compute_hillshade(
    elevation: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    """Return a 0-1 hillshade array from an elevation grid."""
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(elevation)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    shade = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(
        az_rad - aspect
    )
    shade = np.clip(shade, 0, 1)
    shade[np.isnan(elevation)] = np.nan
    return shade


def _read_dem(
    path: Path,
    max_pixels: int = 2048,
) -> tuple[np.ndarray, float, BoundingBox]:
    """Read a DEM GeoTIFF and reproject to Web Mercator for display."""
    with rasterio.open(path) as src:
        nodata_val = src.nodata if src.nodata is not None else -32768.0
        src_crs = src.crs or "EPSG:4326"

        dst_crs = "EPSG:3857"
        transform_3857, w_3857, h_3857 = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )
        width_3857: int = int(cast(int, w_3857))
        height_3857: int = int(cast(int, h_3857))

        longest = max(height_3857, width_3857)
        if longest > max_pixels:
            scale = max_pixels / longest
            width_3857 = max(1, int(width_3857 * scale))
            height_3857 = max(1, int(height_3857 * scale))
            transform_3857, w_3857, h_3857 = calculate_default_transform(
                src_crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
                dst_width=width_3857,
                dst_height=height_3857,
            )
            width_3857 = int(cast(int, w_3857))
            height_3857 = int(cast(int, h_3857))

        dst_array = np.empty((height_3857, width_3857), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform_3857,
            dst_crs=dst_crs,
            dst_nodata=float(nodata_val),
            resampling=Resampling.bilinear,
        )

    dst_array[np.isclose(dst_array, nodata_val)] = np.nan

    merc_left = transform_3857.c
    merc_top = transform_3857.f
    merc_right = merc_left + width_3857 * transform_3857.a
    merc_bottom = merc_top + height_3857 * transform_3857.e

    lon_min, lat_min = _MERCATOR_TO_WGS84.transform(merc_left, merc_bottom)
    lon_max, lat_max = _MERCATOR_TO_WGS84.transform(merc_right, merc_top)

    geo_bounds = BoundingBox(left=lon_min, bottom=lat_min, right=lon_max, top=lat_max)
    return dst_array, nodata_val, geo_bounds


def _render_png(
    elevation: np.ndarray,
    vmin: float,
    vmax: float,
    hillshade: bool = True,
) -> bytes:
    """Render an elevation array to a PNG byte string with transparency."""
    terrain_cmap = _make_terrain_cmap()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    rgba = terrain_cmap(norm(elevation))

    if hillshade:
        shade = _compute_hillshade(elevation)
        shade_blend = 0.35
        for c in range(3):
            rgba[:, :, c] = rgba[:, :, c] * (1 - shade_blend) + shade * shade_blend

    # Transparent where elevation is NaN
    nan_mask = np.isnan(elevation)
    rgba[nan_mask, 3] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = plt.subplots(
            1, 1, figsize=(elevation.shape[1] / 100, elevation.shape[0] / 100), dpi=100
        )
        ax.imshow(rgba, aspect="auto")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf.read()


def _read_flow_acc(
    dem_path: Path,
    max_pixels: int = 2048,
) -> tuple[np.ndarray, BoundingBox]:
    """Compute flow accumulation from *dem_path* and reproject to WGS 84.

    Uses :func:`eoflow.catchment.compute_flow_accumulation` for the
    hydrological conditioning step, then reprojects the resulting array
    to Web Mercator (EPSG:3857) via an in-memory rasterio dataset so the
    same WGS 84 bounds helpers used by :func:`_read_dem` can be reused.

    Parameters
    ----------
    dem_path : Path
        Path to the DEM GeoTIFF.
    max_pixels : int
        Maximum pixels on the longest axis before down-sampling (same
        semantics as in :func:`_read_dem`).

    Returns
    -------
    acc_reprojected : numpy.ndarray
        Float64 array of flow-accumulation values in WGS 84 / Mercator
        projection.  NaN where nodata.
    geo_bounds : rasterio.coords.BoundingBox
        Bounds in WGS 84 (lon_min, lat_min, lon_max, lat_max).
    """
    from rasterio.io import MemoryFile

    acc_array, src_transform, src_crs = compute_flow_accumulation(dem_path)

    height, width = acc_array.shape

    # Write the accumulation array into a MemoryFile so we can use rasterio's
    # reprojection helpers (identical pipeline to _read_dem).
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=src_crs,
            transform=src_transform,
            nodata=float("nan"),
        ) as ds:
            ds.write(acc_array.astype(np.float32), 1)

        with mem.open() as src:
            dst_crs = "EPSG:3857"
            transform_3857, w_3857, h_3857 = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            width_3857 = int(cast(int, w_3857))
            height_3857 = int(cast(int, h_3857))

            longest = max(height_3857, width_3857)
            if longest > max_pixels:
                scale = max_pixels / longest
                width_3857 = max(1, int(width_3857 * scale))
                height_3857 = max(1, int(height_3857 * scale))
                transform_3857, w_3857, h_3857 = calculate_default_transform(
                    src.crs,
                    dst_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                    dst_width=width_3857,
                    dst_height=height_3857,
                )
                width_3857 = int(cast(int, w_3857))
                height_3857 = int(cast(int, h_3857))

            dst_array = np.empty((height_3857, width_3857), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform_3857,
                dst_crs=dst_crs,
                dst_nodata=float("nan"),
                resampling=Resampling.bilinear,
            )

    dst_array = dst_array.astype(np.float64)
    # Values ≤ 0 are nodata (pysheds nodata_out=0 for integer steps)
    dst_array[dst_array <= 0] = np.nan

    merc_left = transform_3857.c
    merc_top = transform_3857.f
    merc_right = merc_left + width_3857 * transform_3857.a
    merc_bottom = merc_top + height_3857 * transform_3857.e

    lon_min, lat_min = _MERCATOR_TO_WGS84.transform(merc_left, merc_bottom)
    lon_max, lat_max = _MERCATOR_TO_WGS84.transform(merc_right, merc_top)

    geo_bounds = BoundingBox(left=lon_min, bottom=lat_min, right=lon_max, top=lat_max)
    return dst_array, geo_bounds


def _render_flow_acc_png(
    acc: np.ndarray,
    threshold: int = 10,
) -> bytes:
    """Render a flow-accumulation array to a log-scale PNG overlay.

    Only cells whose accumulation value exceeds *threshold* are drawn;
    everything else is fully transparent so the base map shows through.
    Rendered with a blue ``cubehelix``-style palette and logarithmic
    normalisation so both small streams and large rivers are visible.

    Parameters
    ----------
    acc : numpy.ndarray
        2-D float array of flow-accumulation values (NaN = nodata).
    threshold : int
        Cells with ``acc <= threshold`` are rendered as transparent.

    Returns
    -------
    bytes
        PNG image bytes (RGBA, with transparency).
    """
    # Mask: show only cells above threshold with valid data
    show_mask = (~np.isnan(acc)) & (acc > threshold)

    if not np.any(show_mask):
        # Return a 1×1 fully-transparent placeholder
        fig, ax = plt.subplots(1, 1, figsize=(0.01, 0.01), dpi=1)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    log_acc = np.where(show_mask, np.log10(np.maximum(acc, 1.0)), np.nan)
    log_min = float(np.nanmin(log_acc))
    log_max = float(np.nanmax(log_acc))

    norm = mcolors.Normalize(vmin=log_min, vmax=log_max if log_max > log_min else log_min + 1)
    cmap = plt.get_cmap("Blues")

    # Map to RGBA; fill non-show cells with zeros first to avoid cmap artefacts
    fill = np.where(np.isnan(log_acc), log_min, log_acc)
    rgba = cmap(norm(fill))

    # Make non-stream cells fully transparent
    rgba[~show_mask, 3] = 0.0

    # Boost opacity of high-accumulation cells so major rivers stand out
    if log_max > log_min:
        stream_alpha = norm(np.where(show_mask, log_acc, log_min))
        rgba[show_mask, 3] = np.clip(0.4 + 0.6 * stream_alpha[show_mask], 0.0, 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = plt.subplots(1, 1, figsize=(acc.shape[1] / 100, acc.shape[0] / 100), dpi=100)
        ax.imshow(rgba, aspect="auto")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# Popup helper
# ---------------------------------------------------------------------------


def _build_popup_html(row: pd.Series, value_col: str) -> str:
    """Return an HTML popup for a sample marker."""
    name = row.get("samplingPoint.prefLabel", "")
    notation = row.get("samplingPoint.notation", "")
    region = row.get("samplingPoint.region", "")
    area = row.get("samplingPoint.area", "")
    time = row.get("phenomenonTime", "")
    value = row.get(value_col, "")
    lat = row.get("latitude", "")
    lon = row.get("longitude", "")
    status = row.get("delineation_status", "")

    raw_acc = row.get("flow_acc_at_pour_point")
    if raw_acc is not None and pd.notna(raw_acc):
        flow_acc_str = f"{int(raw_acc):,} cells"
    else:
        flow_acc_str = ""

    lines = [
        f"<b>{name}</b>",
        f"Notation: {notation}",
        f"Region: {region}",
        f"Area: {area}",
        f"Time: {time}",
        f"<b>{value_col}: {value}</b>",
        f"Delineation: {status}",
        f"Flow accumulation: {flow_acc_str}" if flow_acc_str else "",
        f"Lat: {lat:.5f}, Lon: {lon:.5f}" if isinstance(lat, float) else "",
    ]
    return "<br>".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Core visualisation
# ---------------------------------------------------------------------------


def visualise(
    gpkg_path: Path,
    dem_path: Path | None = None,
    shapefile_path: Path | None = None,
    value_column: str | None = None,
    output_html: Path | None = None,
    *,
    dem_opacity: float = 0.45,
    max_pixels: int = 2048,
    show_flow_acc: bool = True,
    flow_acc_threshold: int = 100,
    flow_acc_opacity: float = 0.7,
) -> Path:
    """Build an interactive map and open it in the browser.

    Parameters
    ----------
    gpkg_path : Path
        GeoPackage produced by ``delineate_catchments.py``.
    dem_path : Path or None
        DEM GeoTIFF covering the study area. If provided, rendered as a
        semi-transparent elevation overlay and used to compute the
        flow-accumulation layer.
    shapefile_path : Path or None
        Boundary shapefile for the study area. If provided, drawn as an
        outline on the map.
    value_column : str or None
        Column to colour-code sample markers by. Auto-detected when
        *None* (defaults to ``result``).
    output_html : Path or None
        Where to write the HTML file. A temporary file is used when *None*.
    dem_opacity : float
        Opacity of the DEM overlay (0 = invisible, 1 = opaque).
    max_pixels : int
        Maximum pixels on the longest DEM axis before down-sampling.
    show_flow_acc : bool
        Whether to compute and display the flow-accumulation overlay.
        Requires *dem_path* to be set.  Defaults to ``True``.
    flow_acc_threshold : int
        Cells with flow accumulation ≤ this value are rendered as
        transparent (i.e. hillslopes are hidden, streams are shown).
        Defaults to ``100``.
    flow_acc_opacity : float
        Overall opacity of the flow-accumulation layer (0–1).
        Defaults to ``0.7``.

    Returns
    -------
    Path
        The path to the written HTML file.
    """
    # ------------------------------------------------------------------
    # 1. Load the GeoPackage
    # ------------------------------------------------------------------
    gdf = gpd.read_file(str(gpkg_path))
    logger.info("Loaded %d features from %s", len(gdf), gpkg_path)

    if gdf.empty:
        raise ValueError("GeoPackage contains no features.")

    # Ensure lat/lon columns exist
    if "latitude" not in gdf.columns or "longitude" not in gdf.columns:
        raise ValueError("GeoPackage must contain 'latitude' and 'longitude' columns.")

    # Determine value column
    value_col = value_column or "result"
    if value_col not in gdf.columns:
        raise ValueError(f"Column '{value_col}' not found. Available: {list(gdf.columns)}")

    gdf["_value"] = pd.to_numeric(gdf[value_col], errors="coerce")
    plot_gdf = gdf.dropna(subset=["latitude", "longitude", "_value"]).copy()

    if plot_gdf.empty:
        raise ValueError("No plottable rows remain after dropping NaN lat/lon/values.")

    # ------------------------------------------------------------------
    # 2. Colour scale for sample values
    # ------------------------------------------------------------------
    vmin_val = float(plot_gdf["_value"].min())
    vmax_val = float(plot_gdf["_value"].max())

    use_log = vmin_val > 0 and vmax_val / vmin_val > 10

    if use_log:
        log_min = math.log10(vmin_val)
        log_max = math.log10(vmax_val)

        def _norm(v: float) -> float:
            if log_max == log_min:
                return 0.5
            return (math.log10(v) - log_min) / (log_max - log_min)

        sample_cmap = cm.LinearColormap(
            colors=["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"],
            vmin=vmin_val,
            vmax=vmax_val,
            caption=f"{value_col} (log scale)",
        )
    else:

        def _norm(v: float) -> float:
            if vmax_val == vmin_val:
                return 0.5
            return (v - vmin_val) / (vmax_val - vmin_val)

        sample_cmap = cm.LinearColormap(
            colors=["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"],
            vmin=vmin_val,
            vmax=vmax_val,
            caption=value_col,
        )

    plot_gdf["_colour"] = plot_gdf["_value"].apply(lambda v: sample_cmap(v))

    # ------------------------------------------------------------------
    # 3. Build the Folium map
    # ------------------------------------------------------------------
    centre_lat = float(plot_gdf["latitude"].mean())
    centre_lon = float(plot_gdf["longitude"].mean())

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
    )

    # ------------------------------------------------------------------
    # 4. DEM overlay (optional)
    # ------------------------------------------------------------------
    if dem_path is not None and Path(dem_path).exists():
        logger.info("Reading DEM from %s …", dem_path)
        try:
            elevation, _nodata, dem_bounds = _read_dem(dem_path, max_pixels=max_pixels)

            valid = elevation[~np.isnan(elevation)]
            if valid.size > 0:
                elev_min = float(np.nanmin(valid))
                elev_max = float(np.nanmax(valid))
                logger.info(
                    "Elevation range: %d – %d m  (%d×%d px)",
                    elev_min,
                    elev_max,
                    elevation.shape[0],
                    elevation.shape[1],
                )

                png_bytes = _render_png(elevation, elev_min, elev_max, hillshade=True)
                png_b64 = base64.b64encode(png_bytes).decode("ascii")
                data_uri = f"data:image/png;base64,{png_b64}"

                img_bounds = [
                    [dem_bounds.bottom, dem_bounds.left],
                    [dem_bounds.top, dem_bounds.right],
                ]
                folium.raster_layers.ImageOverlay(
                    image=data_uri,
                    bounds=img_bounds,
                    opacity=dem_opacity,
                    name="DEM Elevation",
                    interactive=False,
                    cross_origin=False,
                    zindex=1,
                ).add_to(m)

                # Elevation legend
                terrain_colours = [
                    "#1a6e2e",
                    "#5fa843",
                    "#a1c34a",
                    "#ddd07a",
                    "#c4944a",
                    "#9a7250",
                    "#d4c8b8",
                    "#f5f5f5",
                ]
                elev_cmap = cm.LinearColormap(
                    colors=terrain_colours,
                    vmin=elev_min,
                    vmax=elev_max,
                    caption="Elevation (m)",
                )
                elev_cmap.add_to(m)
                logger.info("DEM overlay added (%d KB)", len(png_bytes) // 1024)
        except Exception as exc:
            logger.warning("Could not render DEM overlay: %s", exc)

    # ------------------------------------------------------------------
    # 4b. Flow-accumulation overlay (optional, derived from same DEM)
    # ------------------------------------------------------------------
    if show_flow_acc and dem_path is not None and Path(dem_path).exists():
        logger.info("Computing flow-accumulation overlay from %s …", dem_path)
        try:
            acc_array, acc_bounds = _read_flow_acc(dem_path, max_pixels=max_pixels)

            valid_acc = acc_array[~np.isnan(acc_array)]
            if valid_acc.size > 0:
                acc_max = float(np.nanmax(valid_acc))
                logger.info(
                    "Flow accumulation range: 1 – %.0f cells  (%d×%d px, threshold=%d)",
                    acc_max,
                    acc_array.shape[0],
                    acc_array.shape[1],
                    flow_acc_threshold,
                )

                acc_png_bytes = _render_flow_acc_png(acc_array, threshold=flow_acc_threshold)
                acc_png_b64 = base64.b64encode(acc_png_bytes).decode("ascii")
                acc_data_uri = f"data:image/png;base64,{acc_png_b64}"

                acc_img_bounds = [
                    [acc_bounds.bottom, acc_bounds.left],
                    [acc_bounds.top, acc_bounds.right],
                ]
                folium.raster_layers.ImageOverlay(
                    image=acc_data_uri,
                    bounds=acc_img_bounds,
                    opacity=flow_acc_opacity,
                    name="Flow Accumulation",
                    interactive=False,
                    cross_origin=False,
                    zindex=2,
                ).add_to(m)

                # Add a simple legend entry for the flow-accumulation layer
                acc_cmap = cm.LinearColormap(
                    colors=["#deebf7", "#9ecae1", "#3182bd", "#08306b"],
                    vmin=flow_acc_threshold,
                    vmax=int(acc_max),
                    caption=f"Flow accumulation (cells, log scale, threshold={flow_acc_threshold})",
                )
                acc_cmap.add_to(m)
                logger.info("Flow-accumulation overlay added (%d KB)", len(acc_png_bytes) // 1024)
        except Exception as exc:
            logger.warning("Could not render flow-accumulation overlay: %s", exc)

    # ------------------------------------------------------------------
    # 5. Study area boundary (optional)
    # ------------------------------------------------------------------
    if shapefile_path is not None and Path(shapefile_path).exists():
        logger.info("Loading boundary from %s …", shapefile_path)
        try:
            boundary_gdf = load_shapefile(shapefile_path)
            if boundary_gdf.crs is not None and boundary_gdf.crs != "EPSG:4326":
                boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

            folium.GeoJson(
                boundary_gdf.__geo_interface__,
                name="Boundary",
                style_function=lambda _: {
                    "fillColor": "transparent",
                    "color": "#333333",
                    "weight": 2.5,
                    "dashArray": "6 4",
                    "fillOpacity": 0,
                },
            ).add_to(m)
            logger.info("Boundary added")
        except Exception as exc:
            logger.warning("Could not load boundary: %s", exc)

    # ------------------------------------------------------------------
    # 6. Catchment polygons (hidden by default, revealed on marker click)
    # ------------------------------------------------------------------
    # Each catchment polygon is added directly to the map with `show=False`
    # so Leaflet creates the layer but does not display it initially, and
    # `control=False` so it doesn't clutter the layer-control list. A click
    # handler (added in section 7) toggles the corresponding layer on/off.
    catchment_layer_names: dict[object, str] = {}

    for idx, row in plot_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        val = float(row["_value"])
        colour = str(row["_colour"])
        name = row.get("samplingPoint.prefLabel", "Unknown")

        # Convert the polygon geometry to GeoJSON
        geo_json = gpd.GeoDataFrame([row], geometry="geometry", crs=gdf.crs).__geo_interface__

        catchment_layer = folium.GeoJson(
            geo_json,
            style_function=lambda _, c=colour: {
                "fillColor": c,
                "color": c,
                "weight": 2,
                "fillOpacity": 0.25,
            },
            highlight_function=lambda _, c=colour: {
                "fillColor": c,
                "color": "#000000",
                "weight": 3,
                "fillOpacity": 0.45,
            },
            tooltip=f"{name}: {value_col}={val}",
            show=False,
            control=False,
        )
        catchment_layer.add_to(m)
        catchment_layer_names[idx] = catchment_layer.get_name()

    # ------------------------------------------------------------------
    # 7. Sample point markers and snap-offset lines
    # ------------------------------------------------------------------
    marker_fg = folium.FeatureGroup(name="Sample Points", show=True)
    snap_fg = folium.FeatureGroup(name="Snap Offsets", show=True)

    has_snap_cols = "snap_latitude" in plot_gdf.columns and "snap_longitude" in plot_gdf.columns

    # Pairs of (marker JS var name, catchment JS var name) used to wire up
    # the click-to-reveal behaviour once all layers have been created.
    marker_catchment_pairs: list[tuple[str, str]] = []

    for idx, row in plot_gdf.iterrows():
        val = float(row["_value"])
        colour = str(row["_colour"])
        radius = 5 + 8 * _norm(val)
        popup_html = _build_popup_html(row, value_col)

        lat = float(row["latitude"])
        lon = float(row["longitude"])

        # Sample location marker
        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#333333",
            fill=True,
            fill_color=colour,
            fill_opacity=0.9,
            weight=2,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"{row.get('samplingPoint.prefLabel', '')}: {val} (click for catchment)",
        )
        marker.add_to(marker_fg)

        catchment_name = catchment_layer_names.get(idx)
        if catchment_name is not None:
            marker_catchment_pairs.append((marker.get_name(), catchment_name))

        # Snap-offset line and pour-point marker
        if has_snap_cols:
            snap_lat = row.get("snap_latitude")
            snap_lon = row.get("snap_longitude")
            if pd.notna(snap_lat) and pd.notna(snap_lon):
                snap_lat = float(snap_lat)
                snap_lon = float(snap_lon)

                # Dashed line from sample location to snapped pour point
                folium.PolyLine(
                    locations=[[lat, lon], [snap_lat, snap_lon]],
                    color="#e60000",
                    weight=2,
                    dash_array="5 4",
                    opacity=0.8,
                    tooltip="Snap offset",
                ).add_to(snap_fg)

                # Pour-point marker (small diamond-style marker)
                folium.CircleMarker(
                    location=[snap_lat, snap_lon],
                    radius=4,
                    color="#e60000",
                    fill=True,
                    fill_color="#e60000",
                    fill_opacity=0.9,
                    weight=1,
                    tooltip=f"Pour point: ({snap_lon:.5f}, {snap_lat:.5f})",
                ).add_to(snap_fg)

    marker_fg.add_to(m)
    snap_fg.add_to(m)

    # ------------------------------------------------------------------
    # 7b. Wire up click-to-reveal behaviour for catchment polygons
    # ------------------------------------------------------------------
    if marker_catchment_pairs:
        pairs_js = ",\n            ".join(
            f"[{marker_var}, {catchment_var}]"
            for marker_var, catchment_var in marker_catchment_pairs
        )
        click_js = f"""
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            var targetMap = {m.get_name()};
            var pairs = [
            {pairs_js}
            ];
            var activeCatchment = null;

            function hideActive() {{
                if (activeCatchment) {{
                    targetMap.removeLayer(activeCatchment);
                    activeCatchment = null;
                }}
            }}

            pairs.forEach(function(pair) {{
                var marker = pair[0];
                var catchment = pair[1];
                marker.on('click', function(e) {{
                    if (window.L && L.DomEvent) {{
                        L.DomEvent.stopPropagation(e);
                    }}
                    if (activeCatchment === catchment) {{
                        hideActive();
                        return;
                    }}
                    hideActive();
                    catchment.addTo(targetMap);
                    activeCatchment = catchment;
                }});
            }});

            // Clicking anywhere else on the map hides the active catchment.
            targetMap.on('click', hideActive);
        }});
        </script>
        """
        m.get_root().html.add_child(folium.Element(click_js))

    # ------------------------------------------------------------------
    # 8. Legends and controls
    # ------------------------------------------------------------------
    sample_cmap.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    # Fit map to catchment + point bounds
    all_lats = plot_gdf["latitude"].tolist()
    all_lons = plot_gdf["longitude"].tolist()

    # Include catchment polygon bounds
    catchment_bounds = plot_gdf.geometry.total_bounds  # minx, miny, maxx, maxy
    if not np.any(np.isnan(catchment_bounds)):
        all_lons.extend([catchment_bounds[0], catchment_bounds[2]])
        all_lats.extend([catchment_bounds[1], catchment_bounds[3]])

    sw = [min(all_lats), min(all_lons)]
    ne = [max(all_lats), max(all_lons)]
    m.fit_bounds([sw, ne], padding=[40, 40])

    # --- Opacity sliders (DEM and flow-accumulation layers) ---------------
    if dem_path is not None and Path(dem_path).exists():
        show_acc_slider = show_flow_acc
        slider_html = f"""
        <div style="
            position: fixed; bottom: 50px; left: 10px; z-index: 9999;
            background: white; padding: 10px 14px; border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: sans-serif;
            font-size: 13px; min-width: 200px;
        ">
            <div style="margin-bottom: 6px;">
                <label for="dem-opacity" style="display:inline-block; width:130px;">DEM opacity</label>
                <input type="range" id="dem-opacity" min="0" max="100"
                       value="{
            int(dem_opacity * 100)
        }" style="width: 100px; vertical-align: middle;">
                <span id="dem-opacity-val" style="margin-left: 4px; min-width:32px; display:inline-block;">{
            int(dem_opacity * 100)
        }%</span>
            </div>
            {
            ""
            if not show_acc_slider
            else f'''
            <div>
                <label for="acc-opacity" style="display:inline-block; width:130px;">Flow acc. opacity</label>
                <input type="range" id="acc-opacity" min="0" max="100"
                       value="{int(flow_acc_opacity * 100)}" style="width: 100px; vertical-align: middle;">
                <span id="acc-opacity-val" style="margin-left: 4px; min-width:32px; display:inline-block;">{int(flow_acc_opacity * 100)}%</span>
            </div>
            '''
        }
        </div>
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // DEM slider – targets the FIRST image overlay (z-index 1)
            var demSlider = document.getElementById('dem-opacity');
            var demLabel  = document.getElementById('dem-opacity-val');
            if (demSlider) {{
                demSlider.addEventListener('input', function() {{
                    demLabel.textContent = this.value + '%';
                    var v = this.value / 100;
                    var imgs = document.querySelectorAll('.leaflet-image-layer');
                    if (imgs.length > 0) imgs[0].style.opacity = v;
                }});
            }}
            // Flow-accumulation slider – targets the SECOND image overlay (z-index 2)
            var accSlider = document.getElementById('acc-opacity');
            var accLabel  = document.getElementById('acc-opacity-val');
            if (accSlider) {{
                accSlider.addEventListener('input', function() {{
                    accLabel.textContent = this.value + '%';
                    var v = this.value / 100;
                    var imgs = document.querySelectorAll('.leaflet-image-layer');
                    if (imgs.length > 1) imgs[1].style.opacity = v;
                }});
            }}
        }});
        </script>
        """
        m.get_root().html.add_child(folium.Element(slider_html))

    # ------------------------------------------------------------------
    # 9. Write HTML and open in browser
    # ------------------------------------------------------------------
    if output_html is not None:
        html_path = Path(output_html).resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", prefix="catchments_", delete=False)
        tmp.close()
        html_path = Path(tmp.name).resolve()

    m.save(str(html_path))

    n_catchments = sum(
        1 for _, r in plot_gdf.iterrows() if r.geometry is not None and not r.geometry.is_empty
    )
    logger.info("Map saved to %s", html_path)
    logger.info("  Sample points    : %d", len(plot_gdf))
    logger.info("  Catchments       : %d", n_catchments)
    if dem_path:
        logger.info("  DEM overlay      : %s", dem_path)
    if show_flow_acc and dem_path:
        logger.info("  Flow acc overlay : threshold=%s cells", flow_acc_threshold)
    if shapefile_path:
        logger.info("  Boundary         : %s", shapefile_path)

    webbrowser.open(html_path.as_uri())
    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualise water-quality sample points and their delineated "
            "catchment polygons on an interactive map."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gpkg",
        type=Path,
        required=True,
        help="Path to the GeoPackage produced by delineate_catchments.py.",
    )
    p.add_argument(
        "--dem",
        type=Path,
        default=None,
        help="Path to a DEM GeoTIFF covering the study area. Omit to skip.",
    )
    p.add_argument(
        "--shapefile",
        type=Path,
        default=None,
        help="Path to a boundary shapefile for the study area. Omit to skip.",
    )
    p.add_argument(
        "--value-column",
        type=str,
        default=None,
        help="Column to colour-code markers/catchments by. Auto-detected when omitted.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the HTML map. A temporary file is used when omitted.",
    )
    p.add_argument(
        "--dem-opacity",
        type=float,
        default=0.45,
        help="Opacity of the DEM overlay (0–1).",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=2048,
        help="Max pixels on the longest DEM axis before down-sampling.",
    )
    # Flow-accumulation options
    flow_acc_group = p.add_mutually_exclusive_group()
    flow_acc_group.add_argument(
        "--flow-acc",
        dest="show_flow_acc",
        action="store_true",
        default=True,
        help="Show the flow-accumulation overlay (default: on).",
    )
    flow_acc_group.add_argument(
        "--no-flow-acc",
        dest="show_flow_acc",
        action="store_false",
        help="Disable the flow-accumulation overlay.",
    )
    p.add_argument(
        "--flow-acc-threshold",
        type=int,
        default=100,
        help=(
            "Cells with flow accumulation ≤ this value are hidden "
            "(higher = only show larger streams). Default: 100."
        ),
    )
    p.add_argument(
        "--flow-acc-opacity",
        type=float,
        default=0.7,
        help="Opacity of the flow-accumulation overlay (0–1). Default: 0.7.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.gpkg.exists():
        logger.error("GeoPackage not found: %s", args.gpkg)
        sys.exit(1)

    # Allow user to pass --dem none / --shapefile none to skip
    dem = args.dem if str(args.dem).lower() != "none" else None
    shp = args.shapefile if str(args.shapefile).lower() != "none" else None

    if dem is not None and not dem.exists():
        logger.warning("DEM not found (%s), skipping DEM overlay.", dem)
        dem = None

    if shp is not None and not shp.exists():
        logger.warning("Shapefile not found (%s), skipping boundary.", shp)
        shp = None

    try:
        visualise(
            gpkg_path=args.gpkg,
            dem_path=dem,
            shapefile_path=shp,
            value_column=args.value_column,
            output_html=args.output,
            dem_opacity=args.dem_opacity,
            max_pixels=args.max_pixels,
            show_flow_acc=args.show_flow_acc,
            flow_acc_threshold=args.flow_acc_threshold,
            flow_acc_opacity=args.flow_acc_opacity,
        )
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
