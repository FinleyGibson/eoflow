"""
Folium map visualisation helpers for eoflow.  # noqa: E501

Public API
----------
load_folium_config   Load (or reload) ``folium_config.toml``.
make_base_map        Create a ``folium.Map`` with config defaults.
make_raster_overlay  Warp a BNG ``xarray.DataArray`` to WGS-84 and return
                     a Folium ``ImageOverlay``.
make_raster_overlay_from_tif
                     Reproject a single-band GeoTIFF to WGS-84 and return
                     a Folium ``ImageOverlay``.
legend_html          Generate an HTML colour-bar legend element.
add_area             Add an area boundary polygon (and optional DEM raster)
                     to a Folium map.
add_sample           Add a ``Sample`` — catchment polygon, sample
                     site, pour point, and optional raster overlays — to a
                     Folium map.

Legacy
------
add_polygon_to_folium  Kept for backward compatibility.
"""

from __future__ import annotations

import base64
import io
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import folium
import folium.raster_layers
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize as _MplNormalize
from shapely.geometry import MultiPolygon, Polygon

if TYPE_CHECKING:
    import xarray as xr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

#: Path to the bundled default configuration file.
_CONFIG_PATH: Path = Path(__file__).parent / "folium_config.toml"

#: Module-level config cache.  Populated on first access or after an explicit
#: :func:`load_folium_config` call.
_CONFIG: dict | None = None


def load_folium_config(config_path: Path | str | None = None) -> dict:
    """Load the Folium configuration from a TOML file.

    After calling this function the returned ``dict`` is also cached
    module-globally so all helpers pick up the new values immediately.

    Parameters
    ----------
    config_path:
        Path to a TOML config file.  When ``None`` the bundled
        ``folium_config.toml`` that ships with the package is used.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    global _CONFIG
    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    with open(path, "rb") as fh:
        _CONFIG = tomllib.load(fh)
    return _CONFIG


def _cfg() -> dict:
    """Return the cached config, loading defaults on first access."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_folium_config()
    return _CONFIG


def _get(section: str, key: str, fallback: Any = None) -> Any:
    """Retrieve a single value from the config with a hard-coded fallback."""
    return _cfg().get(section, {}).get(key, fallback)


# ---------------------------------------------------------------------------
# Base map
# ---------------------------------------------------------------------------


def make_base_map(
    location: list[float],
    zoom_start: int | None = None,
    tiles: str | None = None,
    height: int | None = None,
    width: str | None = None,
    scroll_wheel_zoom: bool | None = None,
) -> folium.Map:
    """Create a ``folium.Map`` centred on *location*.

    Every keyword argument falls back to the value in ``folium_config.toml``
    when not supplied explicitly.

    Parameters
    ----------
    location:
        ``[lat, lon]`` map centre in WGS-84 decimal degrees.
    zoom_start:
        Initial zoom level.  Defaults to ``map.zoom_overview`` (9).
    tiles:
        Tile layer name accepted by ``folium.Map(tiles=...)``,
        e.g. ``"CartoDB positron"`` or ``"OpenStreetMap"``.
    height:
        Map height in pixels.  Defaults to ``map.height`` (540).
    width:
        Map width as a CSS string.  Defaults to ``map.width`` ("100%").
    scroll_wheel_zoom:
        Whether scroll-wheel zoom is enabled.  Defaults to
        ``map.scroll_wheel_zoom`` (``False``).

    Returns
    -------
    folium.Map
    """
    return folium.Map(
        location=location,
        zoom_start=(zoom_start if zoom_start is not None else _get("map", "zoom_overview", 9)),
        tiles=(tiles if tiles is not None else _get("map", "tiles", "CartoDB positron")),
        scrollWheelZoom=(
            scroll_wheel_zoom
            if scroll_wheel_zoom is not None
            else _get("map", "scroll_wheel_zoom", False)
        ),
        width=(width if width is not None else _get("map", "width", "100%")),
        height=(height if height is not None else _get("map", "height", 540)),
    )


# ---------------------------------------------------------------------------
# CRS normalisation helper
# ---------------------------------------------------------------------------


def _to_wgs84_geom(polygon_or_gdf, crs: str | None = None):
    """Return a Shapely geometry guaranteed to be in WGS-84 (EPSG:4326).

    Parameters
    ----------
    polygon_or_gdf:
        A :class:`~shapely.geometry.Polygon`,
        :class:`~shapely.geometry.MultiPolygon`, or
        :class:`geopandas.GeoDataFrame` in any CRS.
    crs:
        EPSG string (e.g. ``"EPSG:27700"``) or any identifier accepted by
        :class:`pyproj.CRS` declaring the CRS of *polygon_or_gdf* when it is
        a bare Shapely geometry.  Ignored for ``GeoDataFrame`` inputs (the
        CRS is read from the frame itself).  When ``None`` and the input is
        a bare Shapely geometry it is assumed to already be in WGS-84.

    Returns
    -------
    shapely.geometry.BaseGeometry
        The geometry in EPSG:4326.

    Raises
    ------
    TypeError
        If *polygon_or_gdf* is not a recognised geometry type.
    """
    import geopandas as gpd

    if isinstance(polygon_or_gdf, gpd.GeoDataFrame):
        # Always reproject the whole frame before unioning so that the
        # resulting geometry and its coordinates are in degrees.
        return polygon_or_gdf.to_crs("EPSG:4326").union_all()

    if isinstance(polygon_or_gdf, (Polygon, MultiPolygon)):
        if crs is None:
            # Caller asserts WGS-84 (or we have no information to do better).
            return polygon_or_gdf
        # Wrap in a GeoSeries so pyproj handles the reprojection correctly.
        gs = gpd.GeoSeries([polygon_or_gdf], crs=crs)
        return gs.to_crs("EPSG:4326").iloc[0]

    raise TypeError(
        f"polygon_or_gdf must be a Polygon, MultiPolygon, or GeoDataFrame, "
        f"got {type(polygon_or_gdf).__name__}"
    )


def _geojson_from_input(polygon_or_gdf, crs: str | None = None) -> dict:
    """Return a GeoJSON-compatible dict for *polygon_or_gdf* in WGS-84.

    For a ``GeoDataFrame`` the full feature collection (all rows) is
    preserved rather than collapsing to a single unioned geometry, which
    keeps per-feature attributes available for tooltips / popups.
    """
    import geopandas as gpd
    from shapely.geometry import mapping

    if isinstance(polygon_or_gdf, gpd.GeoDataFrame):
        return polygon_or_gdf.to_crs("EPSG:4326").__geo_interface__

    if isinstance(polygon_or_gdf, (Polygon, MultiPolygon)):
        geom = _to_wgs84_geom(polygon_or_gdf, crs=crs)
        return mapping(geom)  # type: ignore[return-value]

    raise TypeError(
        f"polygon_or_gdf must be a Polygon, MultiPolygon, or GeoDataFrame, "
        f"got {type(polygon_or_gdf).__name__}"
    )


# ---------------------------------------------------------------------------
# Zoom helper
# ---------------------------------------------------------------------------


def _zoom_for_bounds(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    map_width_px: int = 960,
    map_height_px: int = 540,
) -> int:
    """Estimate a Leaflet/Folium zoom level that fits the given WGS-84 bounds.

    Uses the standard Web-Mercator tile formula: at zoom *z* the world is
    ``256 × 2^z`` pixels wide (360° longitude).  The zoom is chosen so that
    the tighter of the two axes (latitude or longitude) fits within the map
    viewport, then reduced by one level to add a comfortable margin.

    Parameters
    ----------
    lat_min, lat_max, lon_min, lon_max:
        Bounding-box corners in decimal degrees (WGS-84).
    map_width_px:
        Assumed map viewport width in pixels, used to size the longitude axis.
    map_height_px:
        Assumed map viewport height in pixels, used to size the latitude axis.

    Returns
    -------
    int
        Zoom level clamped to the range [1, 18].
    """
    import math

    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min

    if lat_span <= 0 or lon_span <= 0:
        return 14  # point / degenerate geometry — use a close-up default

    # At zoom z, 256 × 2^z pixels cover 360° of longitude.
    zoom_lon = math.log2(map_width_px * 360.0 / (256.0 * lon_span))
    # The latitude axis uses 180° as the full extent (equirectangular approx).
    zoom_lat = math.log2(map_height_px * 180.0 / (256.0 * lat_span))

    # Take the more restrictive axis, then back off one level for breathing room.
    zoom = min(zoom_lon, zoom_lat) - 1
    return max(1, min(18, int(zoom)))


# ---------------------------------------------------------------------------
# make_base_map_from_poly
# ---------------------------------------------------------------------------


def make_base_map_from_poly(
    polygon_or_gdf,
    zoom_start: int | None = None,
    tiles: str | None = None,
    height: int | None = None,
    width: str | None = None,
    scroll_wheel_zoom: bool | None = None,
    padding: float = 0.1,
    crs: str | None = None,
) -> folium.Map:
    """Create a ``folium.Map`` automatically centred and zoomed to fit *polygon_or_gdf*.

    Wraps :func:`make_base_map`, computing the map centre from the polygon
    centroid and deriving a zoom level from the bounding-box extent so that
    the geometry fills the viewport without manual tuning.

    The input geometry is **always reprojected to WGS-84** before computing
    the centre and bounds, so you can safely pass a ``GeoDataFrame`` in any
    CRS (e.g. BNG / EPSG:27700) or declare the CRS of a bare Shapely
    geometry via the *crs* parameter.

    Parameters
    ----------
    polygon_or_gdf:
        The geometry to fit.  Accepts a Shapely
        :class:`~shapely.geometry.Polygon`,
        :class:`~shapely.geometry.MultiPolygon`, or a
        :class:`geopandas.GeoDataFrame` (in any CRS — reprojected to
        WGS-84 internally to compute bounds and centroid).
    zoom_start:
        Override the auto-computed zoom level.  When ``None`` (default)
        the zoom is derived from the polygon bounds via
        :func:`_zoom_for_bounds`.
    tiles:
        Forwarded to :func:`make_base_map`.
    height:
        Map height in pixels.  Forwarded to :func:`make_base_map`.
    width:
        Map width CSS string.  Forwarded to :func:`make_base_map`.
    scroll_wheel_zoom:
        Forwarded to :func:`make_base_map`.
    padding:
        Fractional padding added to each side of the bounding box before
        computing the zoom level.  ``0.1`` (10 %) keeps the polygon from
        touching the map edge.  Set to ``0`` for a tight fit.
    crs:
        CRS of *polygon_or_gdf* when it is a bare Shapely geometry, e.g.
        ``"EPSG:27700"`` for BNG.  Ignored for ``GeoDataFrame`` inputs
        (the frame's own CRS is used).  When ``None`` the geometry is
        assumed to already be in WGS-84.

    Returns
    -------
    folium.Map

    Examples
    --------
    Fit a county polygon already in WGS-84:

    .. code-block:: python

        from eoflow.folium import make_base_map_from_poly
        m = make_base_map_from_poly(devon_poly)

    Fit a GeoDataFrame in BNG — reprojected automatically:

    .. code-block:: python

        m = make_base_map_from_poly(devon_gdf_bng)

    Fit a bare BNG Shapely geometry by declaring its CRS:

    .. code-block:: python

        m = make_base_map_from_poly(bng_poly, crs="EPSG:27700")

    Fit a small catchment polygon — auto-zoom picks ~14:

    .. code-block:: python

        m = make_base_map_from_poly(teign.catchment)
    """
    # ── Normalise to a WGS-84 geometry ──────────────────────────────────────
    geom = _to_wgs84_geom(polygon_or_gdf, crs=crs)

    # ── Centre ───────────────────────────────────────────────────────────────
    centre = [geom.centroid.y, geom.centroid.x]

    # ── Zoom ─────────────────────────────────────────────────────────────────
    if zoom_start is None:
        lon_min, lat_min, lon_max, lat_max = geom.bounds

        # Apply padding as a fraction of each span
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min
        lat_min -= lat_span * padding
        lat_max += lat_span * padding
        lon_min -= lon_span * padding
        lon_max += lon_span * padding

        # Use the configured map dimensions to size the zoom calculation
        _raw_width = _get("map", "width", "100%")
        if isinstance(_raw_width, str) and "%" in _raw_width:
            # Percentage widths (e.g. "100%") have no fixed pixel value —
            # fall back to a sensible default for zoom estimation.
            w_px = 960
        elif isinstance(_raw_width, str):
            w_px = int(_raw_width) if _raw_width.isdigit() else 960
        else:
            w_px = int(_raw_width)
        h_px = int(_get("map", "height", 540))

        zoom_start = _zoom_for_bounds(lat_min, lat_max, lon_min, lon_max, w_px, h_px)

    return make_base_map(
        location=centre,
        zoom_start=zoom_start,
        tiles=tiles,
        height=height,
        width=width,
        scroll_wheel_zoom=scroll_wheel_zoom,
    )


# ---------------------------------------------------------------------------
# Raster overlay — BNG xarray.DataArray
# ---------------------------------------------------------------------------


def make_raster_overlay(
    da: "xr.DataArray",
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float | None = None,
    label: str = "Raster",
    percentile_clip: tuple[int, int] | None = None,
) -> folium.raster_layers.ImageOverlay:
    """Warp a BNG (EPSG:27700) ``DataArray`` to WGS-84 and return a
    ``folium.raster_layers.ImageOverlay``.

    BNG is a Transverse Mercator projection so placing an un-warped BNG
    image inside a WGS-84 bounding box introduces a systematic positional
    error.  This function corrects that by back-projecting every WGS-84
    output pixel to fractional BNG array indices and interpolating with a
    bilinear kernel — the same approach used in the ``topography`` and
    ``soil_types`` notebooks.

    Parameters
    ----------
    da:
        2-D ``DataArray`` with dims ``("y", "x")`` in EPSG:27700.
        ``NaN`` values are rendered as transparent pixels.
    cmap:
        Matplotlib colourmap name (e.g. ``"terrain"``, ``"YlOrRd"``).
    vmin, vmax:
        Colour-scale limits.  When ``None`` they are derived from
        *percentile_clip*.
    alpha:
        Overlay opacity (0–1).  Defaults to ``raster.alpha`` in config.
    label:
        Layer name shown in the ``folium.LayerControl``.
    percentile_clip:
        ``(lo, hi)`` percentile pair used to auto-derive *vmin*/*vmax*.
        Defaults to ``(raster.percentile_clip_lo, raster.percentile_clip_hi)``
        from config (2, 98).

    Returns
    -------
    folium.raster_layers.ImageOverlay
    """
    from pyproj import Transformer
    from scipy.ndimage import map_coordinates

    if alpha is None:
        alpha = _get("raster", "alpha", 0.70)
    if percentile_clip is None:
        lo = _get("raster", "percentile_clip_lo", 2)
        hi = _get("raster", "percentile_clip_hi", 98)
        percentile_clip = (lo, hi)

    data = da.values.astype(float)
    xs = da.x.values
    ys = da.y.values
    ny, nx = data.shape
    dx = float(xs[1] - xs[0]) if nx > 1 else 50.0
    dy = float(ys[1] - ys[0]) if ny > 1 else 50.0

    # ── WGS-84 bounding box of the BNG grid corners ─────────────────────────
    tr_fwd = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    cxs = np.array([xs[0], xs[-1], xs[0], xs[-1]])
    cys = np.array([ys[0], ys[0], ys[-1], ys[-1]])
    clons, clats = tr_fwd.transform(cxs, cys)
    lat_min, lat_max = float(clats.min()), float(clats.max())
    lon_min, lon_max = float(clons.min()), float(clons.max())

    # ── Regular WGS-84 output grid (row 0 = northernmost = image top) ────────
    out_lats = np.linspace(lat_max, lat_min, ny)
    out_lons = np.linspace(lon_min, lon_max, nx)
    out_lon_g, out_lat_g = np.meshgrid(out_lons, out_lats)

    # ── Back-project WGS-84 → BNG fractional array indices ──────────────────
    tr_inv = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    bng_x_f, bng_y_f = tr_inv.transform(out_lon_g.ravel(), out_lat_g.ravel())
    row_idx = ((bng_y_f - ys[0]) / dy).reshape(ny, nx)
    col_idx = ((bng_x_f - xs[0]) / dx).reshape(ny, nx)

    # ── NaN-safe bilinear interpolation ──────────────────────────────────────
    valid = np.isfinite(data).astype(float)
    filled = np.where(valid, data, 0.0)
    w_val = map_coordinates(
        filled, [row_idx, col_idx], order=1, mode="constant", cval=0.0, prefilter=False
    )
    w_mask = map_coordinates(
        valid, [row_idx, col_idx], order=1, mode="constant", cval=0.0, prefilter=False
    )
    warped = np.where(w_mask > 0.4, w_val / np.maximum(w_mask, 1e-9), np.nan)

    # ── Auto colour-scale limits ─────────────────────────────────────────────
    finite = warped[np.isfinite(warped)]
    if vmin is None:
        vmin = float(np.percentile(finite, percentile_clip[0])) if finite.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(finite, percentile_clip[1])) if finite.size else 1.0
    vmax = max(vmax, vmin + 1e-6)

    # ── Colourise → RGBA → base64 PNG ────────────────────────────────────────
    norm = _MplNormalize(vmin=vmin, vmax=vmax)
    rgba = (plt.get_cmap(cmap)(norm(warped)) * 255).astype(np.uint8)
    rgba[~np.isfinite(warped), 3] = 0  # transparent outside polygon

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    img_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    return folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=alpha,
        name=label,
    )


# ---------------------------------------------------------------------------
# Raster overlay — GeoTIFF path (any CRS, via rasterio)
# ---------------------------------------------------------------------------


def make_raster_overlay_from_tif(
    tif_path: Path | str,
    cmap_name: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float | None = None,
    name: str = "Raster",
    show: bool = True,
) -> folium.raster_layers.ImageOverlay:
    """Reproject a single-band GeoTIFF to WGS-84 and return a
    ``folium.raster_layers.ImageOverlay``.

    Unlike :func:`make_raster_overlay`, this function accepts any input CRS
    and uses :mod:`rasterio` for reprojection — making it suitable for
    Sentinel-2 products delivered in UTM, or any raster not already in BNG.

    Parameters
    ----------
    tif_path:
        Path to a single-band GeoTIFF in any CRS.
    cmap_name:
        Matplotlib colourmap name.
    vmin, vmax:
        Colour-scale limits.  Auto-derived from the 2nd/98th percentile
        when ``None``.
    alpha:
        Per-pixel opacity applied to valid (non-NaN) pixels (0–1).
        Defaults to ``raster.alpha`` from config.
    name:
        Layer name shown in the ``folium.LayerControl``.
    show:
        Whether the layer is visible by default in the layer control.

    Returns
    -------
    folium.raster_layers.ImageOverlay
    """
    import rasterio
    import rasterio.crs
    import rasterio.warp
    from PIL import Image

    if alpha is None:
        alpha = float(_get("raster", "alpha", 0.70))

    tif_path = Path(tif_path)
    dst_crs = rasterio.crs.CRS.from_epsg(4326)  # type: ignore[attr-defined]

    # ── Reproject to WGS-84 ──────────────────────────────────────────────────
    with rasterio.open(tif_path) as src:
        left, bottom, right, top = src.bounds
        transform_wgs84, w_wgs84, h_wgs84 = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, left, bottom, right, top
        )
        meta = src.meta.copy()
        meta.update(
            crs=dst_crs,
            transform=transform_wgs84,
            width=w_wgs84,
            height=h_wgs84,
            nodata=np.nan,
            dtype="float32",
        )
        mem_path = tif_path.with_stem(tif_path.stem + "__eoflow_wgs84_tmp")
        with rasterio.open(mem_path, "w", **meta) as dst:
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform_wgs84,
                dst_crs=dst_crs,
                resampling=rasterio.warp.Resampling.bilinear,
            )

    with rasterio.open(mem_path) as src_wgs84:
        data = src_wgs84.read(1).astype(np.float32)
        bounds = src_wgs84.bounds  # (left, bottom, right, top) in lon/lat

    mem_path.unlink(missing_ok=True)

    data[~np.isfinite(data)] = np.nan

    # ── Colour-scale limits ──────────────────────────────────────────────────
    v0 = float(vmin) if vmin is not None else float(np.nanpercentile(data, 2))
    v1 = float(vmax) if vmax is not None else float(np.nanpercentile(data, 98))
    norm = _MplNormalize(vmin=v0, vmax=v1, clip=True)
    cmap = plt.get_cmap(cmap_name)

    # ── RGBA → PNG → base64 ──────────────────────────────────────────────────
    rgba = cmap(norm(data), bytes=True)  # (H, W, 4) uint8
    rgba[~np.isfinite(data), 3] = 0
    rgba[np.isfinite(data), 3] = int(alpha * 255)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    folium_bounds = [
        [bounds.bottom, bounds.left],
        [bounds.top, bounds.right],
    ]
    return folium.raster_layers.ImageOverlay(
        image=png_b64,
        bounds=folium_bounds,
        name=name,
        opacity=1.0,
        interactive=False,
        show=show,
    )


# ---------------------------------------------------------------------------
# HTML colour-bar legend
# ---------------------------------------------------------------------------


def legend_html(
    vmin: float,
    vmax: float,
    cmap: str,
    title: str,
    unit: str,
    steps: int | None = None,
) -> str:
    """Return an HTML ``<div>`` colour-bar legend for a raster overlay.

    The legend is absolutely positioned in the bottom-left corner of the
    map viewport and can be injected with::

        f_map.get_root().html.add_child(folium.Element(legend_html(...)))

    Parameters
    ----------
    vmin, vmax:
        Colour-scale limits used to label the bins.
    cmap:
        Matplotlib colourmap name — must match the overlay it describes.
    title:
        Legend title rendered in bold above the swatch list.
    unit:
        Unit label appended to each bin range (e.g. ``"m"``, ``"°"``).
    steps:
        Number of colour bins.  Defaults to ``legend.steps`` in config (5).

    Returns
    -------
    str
        Self-contained HTML ``<div>`` element.
    """
    if steps is None:
        steps = int(_get("legend", "steps", 5))

    edges = np.linspace(vmin, vmax, steps + 1)
    midpts = (edges[:-1] + edges[1:]) / 2

    items: list[str] = []
    for lo, hi, mid in zip(edges[:-1], edges[1:], midpts):
        norm_mid = (mid - vmin) / (vmax - vmin + 1e-12)
        col = mcolors.to_hex(plt.get_cmap(cmap)(norm_mid))
        swatch = (
            f'<span style="background:{col};display:inline-block;'
            "width:12px;height:12px;margin-right:5px;"
            'border:1px solid #888;vertical-align:middle;"></span>'
        )
        items.append(f'<li style="margin:2px 0;">{swatch}{lo:.0f}–{hi:.0f} {unit}</li>')

    rows = "".join(items)
    return (
        '<div style="position:fixed;bottom:28px;left:28px;z-index:1000;'
        "background:white;padding:9px 13px;border-radius:8px;"
        "border:1px solid #ccc;font-size:11px;max-width:210px;"
        'box-shadow:2px 2px 6px rgba(0,0,0,0.15);">'
        f'<b style="font-size:12px;">{title}</b>'
        '<ul style="margin:5px 0 0 0;padding:0;list-style:none;">'
        f"{rows}</ul></div>"
    )


# ---------------------------------------------------------------------------
# add_area
# ---------------------------------------------------------------------------


def add_area(
    polygon_or_gdf,
    f_map: folium.Map | None = None,
    name: str = "Area boundary",
    color: str | None = None,
    fill_color: str | None = None,
    fill_opacity: float | None = None,
    weight: float | None = None,
    dem: "xr.DataArray | None" = None,
    dem_cmap: str | None = None,
    dem_label: str = "Elevation (m)",
    dem_unit: str = "m",
    dem_alpha: float | None = None,
    dem_vmin: float | None = None,
    dem_vmax: float | None = None,
    add_legend: bool = True,
    add_layer_control: bool = False,
    zoom_start: int | None = None,
    crs: str | None = None,
) -> folium.Map:
    """Add an area boundary polygon and an optional DEM raster to a Folium map.

    Typical uses:

    * Draw a county / study-area boundary over a basemap.
    * Overlay a pre-computed elevation, slope, or aspect ``DataArray``
      (EPSG:27700) on top of the boundary.

    The input geometry is **always reprojected to WGS-84** before being
    passed to Folium, so you can safely pass a ``GeoDataFrame`` in any CRS
    (e.g. BNG / EPSG:27700) or declare the CRS of a bare Shapely geometry
    via the *crs* parameter.

    Parameters
    ----------
    polygon_or_gdf:
        The area boundary.  Accepts a Shapely
        :class:`~shapely.geometry.Polygon` /
        :class:`~shapely.geometry.MultiPolygon` or a
        :class:`geopandas.GeoDataFrame` in **any CRS**.
    f_map:
        Existing ``folium.Map`` to add layers to.  When ``None`` a new map
        centred on the geometry centroid is created.
    name:
        Layer name for the boundary ``GeoJson`` layer.
    color:
        Boundary outline colour.  Defaults to ``colors.boundary_color``.
    fill_color:
        Interior fill colour.  Defaults to ``colors.boundary_fill``.
    fill_opacity:
        Interior fill opacity.  Defaults to ``colors.boundary_fill_opacity``.
    weight:
        Outline stroke width in pixels.  Defaults to ``colors.boundary_weight``.
    dem:
        Optional 2-D ``xarray.DataArray`` in EPSG:27700 to render as a
        semi-transparent raster overlay (see :func:`make_raster_overlay`).
    dem_cmap:
        Colourmap for the DEM overlay.  Defaults to ``colormaps.elevation``.
    dem_label:
        Layer name in the ``LayerControl`` for the DEM.
    dem_unit:
        Unit string shown in the legend (e.g. ``"m"`` for elevation,
        ``"°"`` for slope).
    dem_alpha:
        DEM overlay opacity.  Defaults to ``raster.alpha``.
    dem_vmin, dem_vmax:
        Explicit colour-scale limits for the DEM.  Auto-derived from
        config percentiles when ``None``.
    add_legend:
        Add a bottom-left HTML colour-bar legend when *dem* is provided.
    add_layer_control:
        Add a ``folium.LayerControl`` to the map.  Defaults to ``False``
        so that multiple calls to :func:`add_area` / :func:`add_sample`
        can be chained onto the same map before the control is finalised.
        Call ``folium.LayerControl(collapsed=False).add_to(m)`` once at
        the end of your build sequence.
    zoom_start:
        Zoom level used when creating a new map.  Defaults to
        ``map.zoom_overview`` (9).
    crs:
        CRS of *polygon_or_gdf* when it is a bare Shapely geometry, e.g.
        ``"EPSG:27700"`` for BNG.  Ignored for ``GeoDataFrame`` inputs
        (the frame's own CRS is used).  When ``None`` the geometry is
        assumed to already be in WGS-84.

    Returns
    -------
    folium.Map

    Examples
    --------
    Add a WGS-84 county polygon to a new map:

    .. code-block:: python

        m = add_area(devon_poly)

    Add a GeoDataFrame that is still in BNG — reprojected automatically:

    .. code-block:: python

        m = add_area(devon_gdf_bng)

    Add a bare BNG Shapely geometry by declaring its CRS:

    .. code-block:: python

        m = add_area(bng_poly, crs="EPSG:27700")

    Chain with :func:`add_sample` on a shared map, finalise the control
    once at the end:

    .. code-block:: python

        import folium as _folium
        m = make_base_map_from_poly(devon_poly)
        m = add_area(devon_poly, f_map=m)
        m = add_sample(teign, f_map=m)
        _folium.LayerControl(collapsed=False).add_to(m)
    """
    # ── Resolve style defaults ───────────────────────────────────────────────
    color = color or _get("colors", "boundary_color", "#2c6e8a")
    fill_color = fill_color or _get("colors", "boundary_fill", "#d4e8f0")
    fill_opacity = (
        fill_opacity if fill_opacity is not None else _get("colors", "boundary_fill_opacity", 0.15)
    )
    weight = weight if weight is not None else _get("colors", "boundary_weight", 2.5)

    # ── Reproject to WGS-84, derive GeoJSON and centroid ────────────────────
    # _geojson_from_input reprojects GDFs and respects the crs= hint for bare
    # Shapely geometries — guaranteeing that folium always receives degrees.
    geojson = _geojson_from_input(polygon_or_gdf, crs=crs)
    wgs84_geom = _to_wgs84_geom(polygon_or_gdf, crs=crs)
    centre = [wgs84_geom.centroid.y, wgs84_geom.centroid.x]

    # ── Create map if needed ─────────────────────────────────────────────────
    if f_map is None:
        f_map = make_base_map(
            location=centre,
            zoom_start=zoom_start or _get("map", "zoom_overview", 9),
        )

    # ── Boundary vector layer (wrapped in a FeatureGroup so it appears as a
    #    single named entry in the LayerControl) ──────────────────────────────
    area_fg = folium.FeatureGroup(name=name, show=True)
    folium.GeoJson(
        geojson,
        style_function=lambda _: {
            "color": color,
            "fillColor": fill_color,
            "fillOpacity": fill_opacity,
            "weight": weight,
        },
    ).add_to(area_fg)
    area_fg.add_to(f_map)

    # ── Optional DEM raster overlay (added directly so it gets its own
    #    toggle in the LayerControl, independent of the boundary group) ───────
    if dem is not None:
        dem_cmap = dem_cmap or str(_get("colormaps", "elevation", "terrain"))

        overlay = make_raster_overlay(
            dem,
            cmap=dem_cmap,
            vmin=dem_vmin,
            vmax=dem_vmax,
            alpha=dem_alpha,
            label=dem_label,
        )
        overlay.add_to(f_map)

        if add_legend:
            finite = dem.values[np.isfinite(dem.values)]
            lo = _get("raster", "percentile_clip_lo", 2)
            hi = _get("raster", "percentile_clip_hi", 98)
            vmin_leg = (
                dem_vmin
                if dem_vmin is not None
                else (float(np.percentile(finite, lo)) if finite.size else 0.0)
            )
            vmax_leg = (
                dem_vmax
                if dem_vmax is not None
                else (float(np.percentile(finite, hi)) if finite.size else 1.0)
            )
            f_map.get_root().html.add_child(  # type: ignore[union-attr]
                folium.Element(legend_html(vmin_leg, vmax_leg, dem_cmap, dem_label, dem_unit))
            )

    if add_layer_control:
        folium.LayerControl(collapsed=False).add_to(f_map)

    return f_map


# ---------------------------------------------------------------------------
# add_sample
# ---------------------------------------------------------------------------


def add_sample(
    sample,
    f_map: folium.Map | None = None,
    show_catchment: bool = True,
    show_site: bool = True,
    show_pour_point: bool = True,
    show_offset_line: bool = True,
    catchment_color: str | None = None,
    catchment_fill: str | None = None,
    catchment_fill_opacity: float | None = None,
    raster_layers: dict | str | None = None,
    add_layer_control: bool = False,
    zoom_start: int | None = None,
    clip_soil_to_catchment: bool = True,
) -> folium.Map:
    """Add a :class:`~eoflow.samples.Sample` to a Folium map.

    Renders the following elements (each toggleable):

    * **Catchment polygon** — the delineated watershed boundary with a
      metadata popup.
    * **Sample site** — the original EA monitoring point as a red
      ``CircleMarker``.
    * **Pour point** — the DEM-snapped stream outlet as an orange
      ``Marker`` icon.
    * **Offset line** — a dashed ``PolyLine`` connecting the sample site
      to the snapped pour point.
    * **Raster overlays** — spatial layers supplied via *raster_layers*.

    Parameters
    ----------
    sample:
        A :class:`~eoflow.samples.Sample` instance.
    f_map:
        Existing ``folium.Map`` to add layers to.  When ``None`` a new map
        is created centred on the catchment bounding-box midpoint, using
        ``map.zoom_catchment`` from config (14).
    show_catchment:
        Draw the catchment polygon outline and fill.
    show_site:
        Draw the original sampling-point ``CircleMarker``.
    show_pour_point:
        Draw the snapped pour-point ``Marker``.
    show_offset_line:
        Draw the dashed connector between the sample site and the pour
        point (only rendered when both markers are visible).
    catchment_color:
        Catchment outline colour.  Defaults to ``colors.catchment_color``.
    catchment_fill:
        Catchment fill colour.  Defaults to ``colors.catchment_fill``.
    catchment_fill_opacity:
        Catchment fill opacity.  Defaults to
        ``colors.catchment_fill_opacity``.
    raster_layers:
        Spatial raster layers to overlay on the catchment.  Three forms
        are accepted:

        ``"auto"``
            Automatically adds all layers available on ``sample.layers``
            (topography, slope, aspect, soil polygons, and time-mean
            rainfall if computed).

        ``dict``
            A mapping from layer name to one of:

            * A pre-built ``folium.raster_layers.ImageOverlay``.
            * A ``(DataArray, cmap_name)`` tuple — the ``DataArray``
              must be in EPSG:27700; it is warped automatically.
            * A bare ``xarray.DataArray`` in EPSG:27700 (uses the
              elevation colourmap from config).
            * A ``geopandas.GeoDataFrame`` — rendered as a SEARG
              soil-polygon vector layer.

        ``None``
            No raster overlays are added.

    clip_soil_to_catchment:
        When ``True`` (default) the SEARG soil polygons are intersected
        with the catchment boundary before rendering, so only the portion
        of each soil polygon that falls inside the catchment is shown.
        Set to ``False`` to display the full unclipped service polygons.
        Only applies when *raster_layers* is ``"auto"``; pass a
        pre-clipped GeoDataFrame in a dict to control clipping manually.
    add_layer_control:
        Add a ``folium.LayerControl`` to the map.  Defaults to ``False``
        so that multiple calls can be chained onto the same map before the
        control is finalised.  Call
        ``folium.LayerControl(collapsed=False).add_to(m)`` once at the
        end of your build sequence.
    zoom_start:
        Zoom level used when creating a new map.  Defaults to
        ``map.zoom_catchment`` (14).

    Returns
    -------
    folium.Map

    Examples
    --------
    Basic usage — just the vector layers (LayerControl added by caller):

    .. code-block:: python

        import folium as _folium
        from eoflow.folium import add_sample
        m = add_sample(teign)
        _folium.LayerControl(collapsed=False).add_to(m)

    With pre-computed topography and slope from ``sample.layers``:

    .. code-block:: python

        teign.compute_topography(DEM_PATH)
        teign.compute_slope(DEM_PATH)
        m = add_sample(teign, raster_layers="auto")

    With an explicit dict of named overlays:

    .. code-block:: python

        m = add_sample(
            teign,
            raster_layers={
                "Elevation (m)": (topo_da, "terrain"),
                "NDVI":          ndvi_overlay,   # pre-built ImageOverlay
            },
        )
    """
    # ── Resolve style defaults ───────────────────────────────────────────────
    catchment_color = catchment_color or _get("colors", "catchment_color", "#1a1a2e")
    catchment_fill = catchment_fill or _get("colors", "catchment_fill", "#3498db")
    catchment_fill_opacity = (
        catchment_fill_opacity
        if catchment_fill_opacity is not None
        else _get("colors", "catchment_fill_opacity", 0.35)
    )
    catchment_weight = _get("colors", "catchment_weight", 1.2)
    site_color = _get("colors", "sample_point_color", "#c0392b")
    site_fill = _get("colors", "sample_point_fill", "#e74c3c")
    site_radius = _get("colors", "sample_point_radius", 7)
    pp_color = _get("colors", "pour_point_color", "orange")
    line_color = _get("colors", "offset_line_color", "#7f8c8d")
    line_weight = _get("colors", "offset_line_weight", 1.5)
    line_dash = _get("colors", "offset_line_dash", "5 5")

    # ── Create map if needed ─────────────────────────────────────────────────
    if f_map is None:
        if sample.has_catchment:
            bbox = sample.catchment_bbox()  # (W, S, E, N)
            lat_c = (bbox[1] + bbox[3]) / 2
            lon_c = (bbox[0] + bbox[2]) / 2
        else:
            lat_c = sample.latitude
            lon_c = sample.longitude
        f_map = make_base_map(
            location=[lat_c, lon_c],
            zoom_start=zoom_start or _get("map", "zoom_catchment", 14),
        )

    # ── FeatureGroup — all vector elements for this sample travel together ───
    # Using a FeatureGroup means the sample appears as a single named row in
    # the LayerControl, and all its sub-layers persist when f_map is reused.
    sample_fg = folium.FeatureGroup(name=sample.site_name, show=True)

    # ── Catchment polygon ────────────────────────────────────────────────────
    if show_catchment and sample.has_catchment:
        result_str = f"{sample.result} {sample.unit}" if sample.result is not None else "n/a"
        area = sample.catchment_area_km2()
        area_str = f"{area:.3f} km²" if area is not None else "n/a"
        flow_acc = sample.flow_acc_at_pour_point
        flow_str = f"{flow_acc:.0f} cells" if flow_acc is not None else "n/a"
        popup_html = (
            f"<b>{sample.site_name}</b><br>"
            f"Notation : {sample.notation}<br>"
            f"Date     : {sample.date}<br>"
            f"Result   : <b>{result_str}</b><br>"
            f"Area     : {area_str}<br>"
            f"Flow acc : {flow_str}"
        )
        catchment_layer = folium.GeoJson(
            sample.catchment_geojson(),
            style_function=lambda _: {
                "fillColor": catchment_fill,
                "color": catchment_color,
                "weight": catchment_weight,
                "fillOpacity": catchment_fill_opacity,
            },
            tooltip=sample.site_name,
        )
        catchment_layer.add_child(folium.Popup(popup_html, max_width=260))
        catchment_layer.add_to(sample_fg)

    # ── Sample site circle marker ────────────────────────────────────────────
    if show_site and sample.latitude and sample.longitude:
        folium.CircleMarker(
            location=[sample.latitude, sample.longitude],
            radius=site_radius,
            color=site_color,
            fill=True,
            fill_color=site_fill,
            fill_opacity=0.9,
            tooltip=f"Sample site: {sample.site_name}",
            popup=folium.Popup(
                f"<b>{sample.site_name}</b><br>"
                f"lat={sample.latitude:.5f}, lon={sample.longitude:.5f}",
                max_width=220,
            ),
        ).add_to(sample_fg)

    # ── Snapped pour point ───────────────────────────────────────────────────
    if show_pour_point and sample.snap_latitude and sample.snap_longitude:
        folium.Marker(
            location=[sample.snap_latitude, sample.snap_longitude],
            icon=folium.Icon(color=pp_color, icon="tint", prefix="fa"),
            tooltip=(
                f"Pour point — flow acc: {sample.flow_acc_at_pour_point:.0f} cells"
                if sample.flow_acc_at_pour_point is not None
                else "Pour point"
            ),
        ).add_to(sample_fg)

        # Dashed connector between sample site and pour point
        if show_offset_line and sample.latitude and sample.longitude:
            folium.PolyLine(
                locations=[
                    [sample.latitude, sample.longitude],
                    [sample.snap_latitude, sample.snap_longitude],
                ],
                color=line_color,
                weight=line_weight,
                dash_array=line_dash,
            ).add_to(sample_fg)

    # Commit the whole group to the map in one step
    sample_fg.add_to(f_map)

    # ── Raster overlays ──────────────────────────────────────────────────────
    _resolved_layers: dict | None = None
    if raster_layers == "auto":
        _resolved_layers = _layers_from_sample(sample, clip_soil=clip_soil_to_catchment)
    elif isinstance(raster_layers, dict):
        _resolved_layers = raster_layers

    if _resolved_layers:
        try:
            import xarray as xr

            _xr_available = True
        except ImportError:
            xr = None  # type: ignore[assignment]
            _xr_available = False

        try:
            import geopandas as _gpd

            _gpd_available = True
        except ImportError:
            _gpd = None  # type: ignore[assignment]
            _gpd_available = False

        for layer_name, layer in _resolved_layers.items():
            if isinstance(layer, folium.raster_layers.ImageOverlay):
                # Pre-built overlay — add directly
                layer.add_to(f_map)

            elif isinstance(layer, tuple) and len(layer) == 2:
                # (DataArray, cmap_name) — warp BNG → WGS-84
                da, cmap_name = layer
                make_raster_overlay(da, cmap=str(cmap_name), label=layer_name).add_to(f_map)

            elif _xr_available and xr is not None and isinstance(layer, xr.DataArray):
                # Bare DataArray — use elevation colourmap as default
                cmap_name = str(_get("colormaps", "elevation", "terrain"))
                make_raster_overlay(layer, cmap=cmap_name, label=layer_name).add_to(f_map)

            elif _gpd_available and _gpd is not None and isinstance(layer, _gpd.GeoDataFrame):
                # Vector polygon layer (e.g. SEARG soil polygons)
                _make_soil_featuregroup(layer, layer_name).add_to(f_map)

    if add_layer_control:
        folium.LayerControl(collapsed=False).add_to(f_map)

    return f_map


def _make_soil_featuregroup(gdf, name: str) -> folium.FeatureGroup:
    """Build a Folium ``FeatureGroup`` for a SEARG soil-polygon GeoDataFrame.

    Each polygon is coloured using the official EA/DEFRA SEARG colour scheme
    (:data:`eoflow.soil.SEARG_COLOURS`).  Clicking a polygon opens a popup
    with the soil-group name, mapping-unit name, BFI, and SPR values.

    Parameters
    ----------
    gdf:
        GeoDataFrame of SEARG soil polygons in **any CRS** (reprojected to
        WGS-84 internally for Folium).  Expected columns: ``SEARG_Concise``,
        ``MU_NAME``, ``BFI``, ``SPR``, ``SEARGDescription``.
    name:
        Layer name shown in the ``LayerControl``.

    Returns
    -------
    folium.FeatureGroup
        Ready to ``add_to(m)``.
    """
    from eoflow.soil import SEARG_COLOURS

    fg = folium.FeatureGroup(name=name, show=True)

    # Reproject to WGS-84 (Folium requires lat/lon)
    gdf_wgs = gdf.to_crs("EPSG:4326")

    for _, row in gdf_wgs.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        group = row.get("SEARG_Concise") or "Unknown"
        fill = SEARG_COLOURS.get(group, "#aaaaaa")
        bfi = row.get("BFI")
        spr = row.get("SPR")
        desc = str(row.get("SEARGDescription") or "")

        popup_lines = [f"<b>{group}</b>"]
        mu = row.get("MU_NAME")
        if mu:
            popup_lines.append(f"MU: {mu}")
        if bfi is not None:
            try:
                popup_lines.append(f"BFI: {float(bfi):.3f}")
            except (ValueError, TypeError):
                pass
        if spr is not None:
            popup_lines.append(f"SPR: {spr}")
        if desc:
            popup_lines.append(f"<i style='font-size:10px;color:#555'>{desc[:140]}</i>")

        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda _, c=fill: {
                "fillColor": c,
                "color": "#555555",
                "weight": 0.4,
                "fillOpacity": 0.75,
            },
            tooltip=group,
            popup=folium.Popup("<br>".join(popup_lines), max_width=260),
        ).add_to(fg)

    return fg


def _make_index_overlay(
    da: "xr.DataArray",
    *,
    cmap: str,
    label: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
    alpha: float | None = None,
    fallback_bbox: tuple[float, float, float, float] | None = None,
) -> "folium.raster_layers.ImageOverlay | None":
    """Create a Folium :class:`~folium.raster_layers.ImageOverlay` from an
    EO index DataArray (e.g. NDVI, NDWI).

    Collapses the time dimension to a temporal mean, determines the correct
    WGS-84 bounding box from the DataArray's coordinate metadata (with a
    fallback to *fallback_bbox*), and returns a ready-to-use overlay.

    Unlike :func:`make_raster_overlay` this function is not restricted to
    EPSG:27700 (BNG) input — it tries three strategies in order to derive
    WGS-84 bounds from whatever CRS the openEO backend used:

    1. **rioxarray** — if the ``rioxarray`` extension is installed and the
       DataArray carries CRS metadata, the native bounds are reprojected
       via ``pyproj``.
    2. **Degree-range heuristic** — if the ``y``/``lat`` coordinates are
       already in the range [−90, 90] they are treated as decimal degrees.
    3. **Fallback bbox** — the caller-supplied *fallback_bbox* tuple, e.g.
       ``sample.catchment_bbox()``.

    Parameters
    ----------
    da : xarray.DataArray
        Index array, possibly with a time dimension (``"t"`` or ``"time"``).
        All remaining dimensions are assumed to be spatial (``y``/``x`` or
        ``lat``/``lon``).
    cmap : str
        Matplotlib colourmap name (e.g. ``"RdYlGn"`` for NDVI).
    label : str
        Layer name shown in the ``folium.LayerControl``.
    vmin, vmax : float
        Colour-scale limits.  Default is −1 to 1, appropriate for any
        normalised-difference index.
    alpha : float, optional
        Overlay opacity (0–1).  Defaults to ``raster.alpha`` in config.
    fallback_bbox : (W, S, E, N) tuple, optional
        WGS-84 bounding box used when automatic CRS detection fails.
        Pass ``sample.catchment_bbox()`` to fall back to the catchment
        extent when rioxarray is unavailable.

    Returns
    -------
    folium.raster_layers.ImageOverlay or None
        ``None`` when the array cannot be rendered (wrong dimensionality
        after squeezing, or bounds cannot be determined by any strategy).
    """
    if alpha is None:
        alpha = _get("raster", "alpha", 0.70)

    # ── 0. Coerce to float32 ─────────────────────────────────────────────────
    # Some openEO backends (e.g. CDSE / Terrascope) produce object-dtype
    # arrays where NaN is encoded as b'' (empty bytestring).  This survives
    # the NetCDF checkpoint round-trip and makes any numeric reduction crash
    # with "could not convert string to float: b''".
    if not np.issubdtype(da.dtype, np.floating):
        raw = da.values
        if raw.dtype.kind == "O":
            # Vectorised element-wise conversion: b''/''/ None → NaN, rest → float.
            # np.float32(b'0.75') works; Python's float(b'0.75') raises ValueError.
            def _to_f(v: object) -> float:
                if v in (b"", b"nan", "", None):
                    return np.nan
                try:
                    return float(np.float32(v))  # type: ignore[arg-type]
                except (ValueError, TypeError):
                    return np.nan

            da = da.copy(data=np.vectorize(_to_f)(raw).astype(np.float32))
        else:
            try:
                da = da.astype(np.float32)
            except (ValueError, TypeError):
                return None

    # ── 0.5. Drop spurious 'variable' dimension ──────────────────────────────
    # The CDSE / Terrascope openEO backend writes a scalar 'crs' metadata
    # variable alongside the real data variable.  extract_dataarray stacks
    # them into a 'variable' dimension ['crs', 'var'].  Pre-existing
    # checkpoints may still carry this dim; select the first non-CRS slice
    # (or fall back to isel(0)) before any reduction.
    if "variable" in da.dims:
        _META_NAMES = {"crs", "spatial_ref", "crs_wkt"}
        if "variable" in da.coords:
            _data_slices = [
                str(v) for v in da.coords["variable"].values if str(v) not in _META_NAMES
            ]
            da = (
                da.sel(variable=_data_slices[0], drop=True)
                if _data_slices
                else da.isel(variable=0, drop=True)
            )
        else:
            da = da.isel(variable=0, drop=True)

    # ── 1. Collapse the time dimension to a temporal mean ────────────────────
    for time_dim in ("t", "time"):
        if time_dim in da.dims:
            da = da.mean(dim=time_dim, skipna=True)
            break

    # ── 2. Squeeze any remaining length-1 dimensions ─────────────────────────
    da = da.squeeze(drop=True)
    if da.ndim != 2:
        return None

    # ── 3. Determine WGS-84 bounding box ─────────────────────────────────────
    lat_min = lat_max = lon_min = lon_max = None

    # Strategy A: rioxarray CRS metadata + pyproj reprojection
    try:
        import rioxarray  # noqa: F401
        from pyproj import Transformer

        if da.rio.crs is not None:
            epsg = da.rio.crs.to_epsg()
            if epsg is not None:
                w, s, e, n = da.rio.bounds()  # bounds in native CRS
                tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
                xs_c = np.array([w, e, w, e])
                ys_c = np.array([s, s, n, n])
                lons, lats = tr.transform(xs_c, ys_c)
                lat_min, lat_max = float(lats.min()), float(lats.max())
                lon_min, lon_max = float(lons.min()), float(lons.max())
    except Exception:
        pass

    # Strategy B: coordinate values already in degree range → treat as WGS-84
    if lat_min is None:
        for y_name in ("latitude", "lat", "y"):
            if y_name in da.coords:
                y_vals = da.coords[y_name].values.ravel()
                if float(y_vals.min()) >= -90.0 and float(y_vals.max()) <= 90.0:
                    lat_min, lat_max = float(y_vals.min()), float(y_vals.max())
                    break
        for x_name in ("longitude", "lon", "x"):
            if x_name in da.coords:
                x_vals = da.coords[x_name].values.ravel()
                if float(x_vals.min()) >= -180.0 and float(x_vals.max()) <= 360.0:
                    lon_min, lon_max = float(x_vals.min()), float(x_vals.max())
                    break

    # Strategy C: caller-supplied fallback bbox (e.g. catchment WGS-84 bbox)
    if lat_min is None and fallback_bbox is not None:
        lon_min, lat_min, lon_max, lat_max = fallback_bbox  # (W, S, E, N)

    if lat_min is None or lon_min is None or lat_max is None or lon_max is None:
        return None

    # Narrow to plain floats so type-checkers (and ImageOverlay) are satisfied.
    _lat_min: float = float(lat_min)
    _lat_max: float = float(lat_max)
    _lon_min: float = float(lon_min)
    _lon_max: float = float(lon_max)
    _alpha: float = float(alpha) if alpha is not None else float(_get("raster", "alpha", 0.70))

    # ── 4. Extract array values and ensure north-up orientation ──────────────
    data = da.values.astype(float)

    # Auto-adjust colour limits when the data falls outside [vmin, vmax].
    # CDSE / Terrascope may return unscaled values (e.g. NDVI × 1000) rather
    # than the conventional −1 … 1 range.  Use 2nd/98th percentiles rather
    # than absolute min/max so that a handful of outlier pixels cannot
    # compress the entire real distribution into a single colour band.
    finite = data[np.isfinite(data)]
    if finite.size > 0:
        _dmin, _dmax = float(finite.min()), float(finite.max())
        if _dmin < vmin or _dmax > vmax:
            vmin = float(np.percentile(finite, 2))
            vmax = float(np.percentile(finite, 98))
    # openEO backends often store y ascending (south→north).  Folium expects
    # row 0 at the top (north) edge of the bounds, so flip when necessary.
    for y_name in ("latitude", "lat", "y"):
        if y_name in da.dims and y_name in da.coords:
            y_vals = da.coords[y_name].values
            if len(y_vals) > 1 and float(y_vals[0]) < float(y_vals[-1]):
                data = data[::-1, :]
            break

    # ── 5. Colourise → RGBA → base64 PNG ─────────────────────────────────────
    norm = _MplNormalize(vmin=vmin, vmax=vmax)
    rgba = (plt.get_cmap(cmap)(norm(data)) * 255).astype(np.uint8)
    # Make NaN pixels fully transparent
    rgba[~np.isfinite(da.values.astype(float)), 3] = 0

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    buf.seek(0)
    img_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    return folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[_lat_min, _lon_min], [_lat_max, _lon_max]],
        opacity=_alpha,
        name=label,
    )


def _layers_from_sample(sample, *, clip_soil: bool = True) -> dict:
    """Build a ``raster_layers`` dict from ``sample.layers``.

    Called internally by :func:`add_sample` when ``raster_layers="auto"``.

    For each available layer the appropriate colourmap is taken from the
    loaded config:

    * ``topography``  → ``colormaps.elevation``
    * ``slope``       → ``colormaps.slope``
    * ``aspect``      → ``colormaps.aspect``
    * ``soil_polygons`` → GeoDataFrame (optionally clipped to catchment)
    * ``rainfall``    → ``colormaps.rainfall`` (collapsed to time-mean)

    The ``soil_type`` layer is a ``pandas.Series`` (fractional coverage),
    not a raster, so it is excluded here.

    Parameters
    ----------
    sample:
        A :class:`~eoflow.samples.Sample` with a populated
        ``.layers`` attribute.
    clip_soil:
        When ``True`` (default) the soil polygons GeoDataFrame is
        intersected with the sample's catchment boundary before being
        added to the returned dict.  Polygons that become empty after
        clipping are dropped.  Has no effect when the sample has no
        catchment or no ``soil_polygons`` layer.

    Returns
    -------
    dict
        ``{layer_name: value}`` where value is either a
        ``(DataArray, cmap_name)`` tuple (raster layers) or a
        ``geopandas.GeoDataFrame`` (soil polygon layer).
    """
    layers: dict = {}
    sl = getattr(sample, "layers", None)
    if sl is None:
        return layers

    if getattr(sl, "topography", None) is not None:
        layers["Elevation (m)"] = (
            sl.topography,
            _get("colormaps", "elevation", "terrain"),
        )

    if getattr(sl, "slope", None) is not None:
        unit = sl.slope.attrs.get("units", "°")
        layers[f"Slope ({unit})"] = (
            sl.slope,
            _get("colormaps", "slope", "YlOrRd"),
        )

    if getattr(sl, "aspect", None) is not None:
        unit = sl.aspect.attrs.get("units", "°")
        layers[f"Aspect ({unit})"] = (
            sl.aspect,
            _get("colormaps", "aspect", "twilight"),
        )

    if getattr(sl, "soil_polygons", None) is not None and not sl.soil_polygons.empty:
        soil_gdf = sl.soil_polygons
        if clip_soil and getattr(sample, "catchment", None) is not None:
            try:
                import geopandas as _gpd

                # Reproject catchment from WGS-84 to the GDF's native BNG CRS
                catchment_bng = (
                    _gpd.GeoSeries([sample.catchment], crs="EPSG:4326").to_crs("EPSG:27700").iloc[0]
                )
                clipped = soil_gdf.copy()
                clipped["geometry"] = soil_gdf.geometry.intersection(catchment_bng)
                clipped = clipped[
                    clipped.geometry.notna() & ~clipped.geometry.is_empty
                ].reset_index(drop=True)
                soil_gdf = clipped
            except Exception:
                pass  # fall back to unclipped on any error
        layers["SEARG Soil Types"] = soil_gdf

    if getattr(sl, "rainfall", None) is not None:
        # Collapse the time dimension before rendering
        rain_mean = sl.rainfall.mean(dim="time", skipna=True)
        layers["Rainfall (mm/h, time-mean)"] = (
            rain_mean,
            _get("colormaps", "rainfall", "Blues"),
        )

    # ── EO spectral indices ───────────────────────────────────────────────────
    # NDVI and NDWI are time-series DataArrays from openEO (not in BNG), so
    # they are handled by _make_index_overlay rather than make_raster_overlay.
    _catchment_bbox = sample.catchment_bbox() if hasattr(sample, "catchment_bbox") else None

    if getattr(sl, "ndvi", None) is not None:
        overlay = _make_index_overlay(
            sl.ndvi,
            cmap=_get("colormaps", "ndvi", "RdYlGn"),
            label="NDVI (mean)",
            fallback_bbox=_catchment_bbox,
        )
        if overlay is not None:
            layers["NDVI (mean)"] = overlay

    if getattr(sl, "ndwi", None) is not None:
        overlay = _make_index_overlay(
            sl.ndwi,
            cmap=_get("colormaps", "ndwi", "RdBu"),
            label="NDWI (mean)",
            fallback_bbox=_catchment_bbox,
        )
        if overlay is not None:
            layers["NDWI (mean)"] = overlay

    return layers


# ---------------------------------------------------------------------------
# Legacy helper  (kept for backward compatibility)
# ---------------------------------------------------------------------------


def add_polygon_to_folium(
    polygon: Polygon | MultiPolygon,
    f_map=None,
    color="blue",
    weight=3,
    fill=True,
    fill_opacity=0.4,
    map_kwargs={},
):
    """Add a Shapely Polygon (or MultiPolygon) to a Folium map.

    .. deprecated::
        Prefer :func:`add_area` for new code.  This function is retained
        for backward compatibility.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or MultiPolygon
        The geometry to add.
    f_map : folium.Map or None
        Existing map object. If None, a new map will be created
        centered on the polygon.
    color : str
        Outline color.
    weight : int
        Line weight.
    fill : bool
        Whether to fill the polygon.
    fill_opacity : float
        Fill opacity.

    Returns
    -------
    folium.Map
        The resulting map object.
    """
    # If no map supplied, initialize centered on polygon centroid
    if f_map is None:
        f_map = folium.Map(
            location=[polygon.centroid.y, polygon.centroid.x],
            zoom_start=13,
            **map_kwargs,
        )

    # Handle MultiPolygon recursively
    if isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            f_map = add_polygon_to_folium(
                poly,
                f_map=f_map,
                color=color,
                weight=weight,
                fill=fill,
                fill_opacity=fill_opacity,
            )
    elif isinstance(polygon, Polygon):
        # Extract exterior coordinates
        exterior_coords = [(y, x) for x, y in polygon.exterior.coords]  # folium uses (lat, lon)

        # Create folium polygon layer
        folium.Polygon(
            locations=exterior_coords,
            color=color,
            weight=weight,
            fill=fill,
            fill_opacity=fill_opacity,
        ).add_to(f_map)

        # Add holes, if any
        for interior in polygon.interiors:
            interior_coords = [(y, x) for x, y in interior.coords]
            folium.Polygon(
                locations=interior_coords,
                color=color,
                weight=weight,
                fill=False,  # holes are outlines only
            ).add_to(f_map)
    else:  # noqa: B901
        raise ValueError(f"Unsupported geometry type: {type(polygon)}")

    return f_map
