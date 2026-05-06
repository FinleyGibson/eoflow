import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from eoflow.folium import add_area, add_sample, make_base_map_from_poly
from eoflow.log_utils import get_logger
from eoflow.samples import Sample
from eoflow.utils import DATA_DIR, PROJECT_ROOT

# 0 Set up logging
logger = get_logger("eoflow.scripts.full_sample_flow")

# 1. Load Required Data


# ---------------------------------------------------------------------------
# Layer visualisation helper
# ---------------------------------------------------------------------------


def _plot_sample_layers(
    cs: Sample,
    output_path: Path,
    *,
    dem_path: Path | None = None,
) -> None:
    """Render a sequence of subplots — catchment boundary + each computed
    layer — and save the figure to *output_path* as a PNG.

    Panels produced (when the layer is available):

    1. Catchment — polygon with flow-accumulation background (WGS-84)
    2. Elevation  — topography DataArray (BNG, m)
    3. Slope      — slope DataArray (BNG, degrees / %)
    4. Aspect     — aspect DataArray (BNG, cyclic 0–360 °)
    5. SEARG soil — soil_polygons GeoDataFrame with official EA colours
    6. Rainfall   — time-mean of rainfall DataArray (BNG, mm/h)
    7. NDVI       — time-mean of NDVI DataArray (any CRS → shown in WGS-84)

    Each panel has a CartoDB Positron basemap tile layer as background.
    Raster/polygon data are drawn semi-transparently so the basemap
    remains visible.  Each raster panel carries a colourbar; the soil
    panel carries a colour-patch legend.  The catchment outline is drawn
    as a dark border over every panel.  The original sample point (red
    circle) and snapped pour point (blue triangle) are marked on every
    panel.

    Parameters
    ----------
    cs : Sample
        A fully populated sample (catchment + layers).
    output_path : Path
        Destination PNG file.
    dem_path : Path, optional
        Path to the DEM GeoTIFF.  When supplied, flow accumulation is
        computed and displayed in the catchment panel.
    """
    from pyproj import Transformer

    from eoflow.soil import SEARG_COLOURS

    if not cs.has_catchment:
        logger.warning("Sample has no catchment polygon — skipping layer plot.")
        return

    catchment_wgs84 = cs.catchment  # Shapely geometry in EPSG:4326
    assert catchment_wgs84 is not None  # guaranteed by has_catchment check above

    # Reproject catchment to BNG for overlaying on BNG rasters / soil.
    catchment_bng_gs = gpd.GeoSeries([catchment_wgs84], crs="EPSG:4326").to_crs("EPSG:27700")  # type: ignore[call-overload]
    catchment_bng = catchment_bng_gs.iloc[0]
    catchment_wgs84_gs = gpd.GeoSeries([catchment_wgs84], crs="EPSG:4326")  # type: ignore[call-overload]

    bbox_bng = catchment_bng.bounds  # type: ignore[union-attr]  # (minx, miny, maxx, maxy) BNG metres
    bbox_wgs = catchment_wgs84.bounds  # (minx, miny, maxx, maxy) WGS-84

    # Transformer for reprojecting WGS-84 sample/pour points to BNG.
    tr_to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------

    def _bng_buf(frac: float = 0.15):
        """Return (xlo, xhi, ylo, yhi) BNG limits with fractional padding."""
        dx = (bbox_bng[2] - bbox_bng[0]) * frac
        dy = (bbox_bng[3] - bbox_bng[1]) * frac
        return bbox_bng[0] - dx, bbox_bng[2] + dx, bbox_bng[1] - dy, bbox_bng[3] + dy

    def _wgs_buf(frac: float = 0.15):
        """Return (xlo, xhi, ylo, yhi) WGS-84 limits with fractional padding."""
        dx = (bbox_wgs[2] - bbox_wgs[0]) * frac
        dy = (bbox_wgs[3] - bbox_wgs[1]) * frac
        return bbox_wgs[0] - dx, bbox_wgs[2] + dx, bbox_wgs[1] - dy, bbox_wgs[3] + dy

    def _km_fmt(val: float, _) -> str:
        """Format BNG metre values as compact kilometre strings, e.g. '250k'."""
        return f"{val / 1000:.0f}k"

    def _boundary_bng(ax: plt.Axes) -> None:  # type: ignore[name-defined]
        """Draw the catchment boundary (BNG) on *ax*."""
        from shapely.geometry import MultiPolygon

        polys = (
            list(catchment_bng.geoms)  # type: ignore[union-attr]
            if isinstance(catchment_bng, MultiPolygon)
            else [catchment_bng]
        )
        for poly in polys:
            xs, ys = poly.exterior.xy  # type: ignore[union-attr]
            ax.plot(xs, ys, color="#1a1a2e", linewidth=1.8, zorder=8)
            for ring in poly.interiors:  # type: ignore[union-attr]
                xi, yi = ring.xy
                ax.plot(xi, yi, color="#1a1a2e", linewidth=1.0, zorder=8)

    def _boundary_wgs84(ax: plt.Axes) -> None:  # type: ignore[name-defined]
        """Draw the catchment boundary (WGS-84) on *ax*."""
        from shapely.geometry import MultiPolygon

        polys = (
            list(catchment_wgs84.geoms)
            if isinstance(catchment_wgs84, MultiPolygon)
            else [catchment_wgs84]
        )
        for poly in polys:
            xs, ys = poly.exterior.xy  # type: ignore[union-attr]
            ax.plot(xs, ys, color="#1a1a2e", linewidth=1.8, zorder=8)
            for ring in poly.interiors:  # type: ignore[union-attr]
                xi, yi = ring.xy
                ax.plot(xi, yi, color="#1a1a2e", linewidth=1.0, zorder=8)

    def _style_bng_ax(ax: plt.Axes, title: str) -> None:  # type: ignore[name-defined]
        """Apply shared BNG axis styling (labels, km tick formatter, title)."""
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("Easting (m)", fontsize=7)
        ax.set_ylabel("Northing (m)", fontsize=7)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_km_fmt))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_km_fmt))
        ax.tick_params(labelsize=7)

    def _mark_points_bng(ax: plt.Axes, *, legend: bool = False) -> None:  # type: ignore[name-defined]
        """Plot the sample point and snapped pour point on a BNG axes."""
        handles = []
        sp = cs.sample_point
        if sp is not None:
            x_b, y_b = tr_to_bng.transform(sp.x, sp.y)
            h = ax.plot(
                x_b,
                y_b,
                marker="o",
                color="#e74c3c",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=10,
                linestyle="none",
                label="Sample point",
            )[0]
            handles.append(h)
        pp = cs.snapped_pour_point
        if pp is not None:
            x_b, y_b = tr_to_bng.transform(pp.x, pp.y)
            h = ax.plot(
                x_b,
                y_b,
                marker="^",
                color="#2980b9",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=10,
                linestyle="none",
                label="Pour point",
            )[0]
            handles.append(h)
        if legend and handles:
            ax.legend(
                handles=handles,
                loc="upper right",
                fontsize=6,
                framealpha=0.85,
                borderpad=0.4,
            )

    def _mark_points_wgs84(ax: plt.Axes, *, legend: bool = False) -> None:  # type: ignore[name-defined]
        """Plot the sample point and snapped pour point on a WGS-84 axes."""
        handles = []
        sp = cs.sample_point
        if sp is not None:
            h = ax.plot(
                sp.x,
                sp.y,
                marker="o",
                color="#e74c3c",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=10,
                linestyle="none",
                label="Sample point",
            )[0]
            handles.append(h)
        pp = cs.snapped_pour_point
        if pp is not None:
            h = ax.plot(
                pp.x,
                pp.y,
                marker="^",
                color="#2980b9",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=10,
                linestyle="none",
                label="Pour point",
            )[0]
            handles.append(h)
        if legend and handles:
            ax.legend(
                handles=handles,
                loc="upper right",
                fontsize=6,
                framealpha=0.85,
                borderpad=0.4,
            )

    def _add_basemap(ax: plt.Axes, crs_str: str) -> None:  # type: ignore[name-defined]
        """Fetch CartoDB Positron tiles and paint them behind the data.

        Axis limits *must* already be set before calling this so that
        contextily knows which tiles to download.  Silently skipped when
        offline or if contextily raises for any other reason.
        """
        try:
            import contextily as ctx  # type: ignore[import-untyped]

            ctx.add_basemap(
                ax,
                crs=crs_str,
                source=ctx.providers.CartoDB.Positron,
                attribution=False,
                zoom="auto",
            )
        except Exception:
            pass  # offline or package unavailable — no basemap, no crash

    def _render_bng_raster(
        ax: plt.Axes,  # type: ignore[name-defined]
        da,
        title: str,
        cmap: str,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        cyclic: bool = False,
    ) -> None:
        """imshow a BNG DataArray (y ascending south→north) with a colourbar.

        Sets axis limits and fetches the basemap *before* drawing the
        raster so the tiles are visible through the semi-transparent overlay.
        """
        vals = da.values.astype(float)
        x_c = da.coords["x"].values
        y_c = da.coords["y"].values

        # y is ascending (south → north); origin='lower' places row 0 at the
        # bottom of the image, matching the southernmost y value.
        extent = (
            float(x_c.min()),
            float(x_c.max()),
            float(y_c.min()),
            float(y_c.max()),
        )

        finite = vals[np.isfinite(vals)]
        if cyclic:
            vmin_use, vmax_use = 0.0, 360.0
        elif vmin is not None and vmax is not None:
            vmin_use, vmax_use = vmin, vmax
        elif finite.size > 0:
            vmin_use = float(np.percentile(finite, 2))
            vmax_use = float(np.percentile(finite, 98))
            if vmin_use == vmax_use:
                vmin_use, vmax_use = float(finite.min()), float(finite.max())
        else:
            vmin_use, vmax_use = 0.0, 1.0

        # ── Set limits then fetch tiles (order matters for contextily) ──
        xlo, xhi, ylo, yhi = _bng_buf()
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        _style_bng_ax(ax, title)
        _add_basemap(ax, "EPSG:27700")

        im = ax.imshow(
            vals,
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=vmin_use,
            vmax=vmax_use,
            aspect="auto",
            interpolation="nearest",
            alpha=0.75,
            zorder=2,
        )
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03, fraction=0.046)
        _boundary_bng(ax)

    # ------------------------------------------------------------------
    def _render_eo_raster(ax: plt.Axes, da: Any, title: str, cmap: str) -> None:  # type: ignore[name-defined]
        """Render an openEO DataArray (any CRS) clipped to the catchment.

        Handles WGS-84, BNG, and UTM projections by always reprojecting
        image corners to WGS-84 for display.  Applies the catchment mask
        before the y-flip so the rasterio north-up convention is respected.
        Adds a CartoDB basemap and a colourbar.
        """
        # ── Coerce object-dtype (openEO backend quirk) ───────────────────
        if not np.issubdtype(da.dtype, np.floating):
            raw = da.values
            if raw.dtype.kind == "O":

                def _to_f_eo(v: object) -> float:
                    if v in (b"", b"nan", "", None):
                        return np.nan
                    try:
                        return float(np.float32(v))  # type: ignore[arg-type]
                    except (ValueError, TypeError):
                        return np.nan

                da = da.copy(data=np.vectorize(_to_f_eo)(raw).astype(np.float32))
            else:
                try:
                    da = da.astype(np.float32)
                except Exception:
                    ax.set_visible(False)
                    return

        # ── Drop spurious 'variable' dimension ───────────────────────────
        if "variable" in da.dims:
            _META = {"crs", "spatial_ref", "crs_wkt"}
            if "variable" in da.coords:
                data_slices = [str(v) for v in da.coords["variable"].values if str(v) not in _META]
                da = (
                    da.sel(variable=data_slices[0], drop=True)
                    if data_slices
                    else da.isel(variable=0, drop=True)
                )
            else:
                da = da.isel(variable=0, drop=True)

        # ── Collapse time to mean ─────────────────────────────────────────
        for time_dim in ("t", "time"):
            if time_dim in da.dims:
                da = da.mean(dim=time_dim, skipna=True)
                break
        da = da.squeeze(drop=True)

        if da.ndim != 2:
            ax.set_visible(False)
            return

        # ── Resolve spatial coordinate arrays ────────────────────────────
        y_c = np.linspace(0.0, 1.0, da.shape[0])
        for yn in ("latitude", "lat", "y"):
            if yn in da.coords:
                y_c = da.coords[yn].values
                break
        x_c = np.linspace(0.0, 1.0, da.shape[1])
        for xn in ("longitude", "lon", "x"):
            if xn in da.coords:
                x_c = da.coords[xn].values
                break

        x_min_v = float(x_c.min())
        x_max_v = float(x_c.max())
        y_min_v = float(y_c.min())
        y_max_v = float(y_c.max())

        # ── Determine CRS and derive WGS-84 image extent ─────────────────
        # Case A: y already in degree range → WGS-84
        # Case B: BNG (x & y both in [0, 800 000]) → reproject via pyproj
        # Case C: anything else (UTM etc.) → infer UTM zone from centroid
        if -90.0 <= y_min_v and y_max_v <= 90.0:
            wgs_x_min, wgs_x_max = x_min_v, x_max_v
            wgs_y_min, wgs_y_max = y_min_v, y_max_v
        elif 0.0 <= y_min_v and y_max_v <= 800_000.0 and 0.0 <= x_min_v and x_max_v <= 800_000.0:
            tr_b = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
            c_lon, c_lat = tr_b.transform(
                [x_min_v, x_max_v, x_min_v, x_max_v],
                [y_min_v, y_min_v, y_max_v, y_max_v],
            )
            wgs_x_min, wgs_x_max = float(min(c_lon)), float(max(c_lon))
            wgs_y_min, wgs_y_max = float(min(c_lat)), float(max(c_lat))
        else:
            _clon = catchment_wgs84.centroid.x
            _clat = catchment_wgs84.centroid.y
            _zone = int((_clon + 180.0) / 6.0) + 1
            _epsg = 32600 + _zone if _clat >= 0.0 else 32700 + _zone
            tr_u = Transformer.from_crs(f"EPSG:{_epsg}", "EPSG:4326", always_xy=True)
            c_lon, c_lat = tr_u.transform(
                [x_min_v, x_max_v, x_min_v, x_max_v],
                [y_min_v, y_min_v, y_max_v, y_max_v],
            )
            wgs_x_min, wgs_x_max = float(min(c_lon)), float(max(c_lon))
            wgs_y_min, wgs_y_max = float(min(c_lat)), float(max(c_lat))

        # ── Mask outside catchment BEFORE y-flip ─────────────────────────
        # rasterio.from_bounds is north-up (row 0 = north), matching the
        # raw DataArray when y is descending.  Masking after the flip would
        # invert the visible region.
        vals = da.values.astype(float)
        try:
            import rasterio.features
            import rasterio.transform

            H, W = vals.shape
            affine = rasterio.transform.from_bounds(
                wgs_x_min, wgs_y_min, wgs_x_max, wgs_y_max, W, H
            )
            outside = rasterio.features.geometry_mask(
                [catchment_wgs84.__geo_interface__],
                out_shape=(H, W),
                transform=affine,
                invert=False,
            )
            vals[outside] = np.nan
        except Exception as exc:
            logger.debug("EO raster catchment masking failed (%s) — full extent shown.", exc)

        # ── Flip to south-up for origin='lower' ──────────────────────────
        if len(y_c) > 1 and float(y_c[0]) > float(y_c[-1]):
            vals = vals[::-1, :]

        # ── Auto-scale colour limits (2nd–98th percentile) ───────────────
        finite = vals[np.isfinite(vals)]
        vmin_use, vmax_use = -1.0, 1.0
        if finite.size > 0:
            _dmin, _dmax = float(finite.min()), float(finite.max())
            if _dmin < vmin_use or _dmax > vmax_use:
                vmin_use = float(np.percentile(finite, 2))
                vmax_use = float(np.percentile(finite, 98))

        wgs_extent = (wgs_x_min, wgs_x_max, wgs_y_min, wgs_y_max)

        # ── Limits → basemap → image ──────────────────────────────────────
        xlo, xhi, ylo, yhi = _wgs_buf()
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("Longitude (°)", fontsize=7)
        ax.set_ylabel("Latitude (°)", fontsize=7)
        ax.tick_params(labelsize=7)
        _add_basemap(ax, "EPSG:4326")

        cmap_masked = plt.get_cmap(cmap).copy()
        cmap_masked.set_bad(alpha=0.0)

        im = ax.imshow(
            vals,
            extent=wgs_extent,
            origin="lower",
            cmap=cmap_masked,
            vmin=vmin_use,
            vmax=vmax_use,
            aspect="auto",
            interpolation="nearest",
            alpha=0.85,
            zorder=2,
        )
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03, fraction=0.046)
        _boundary_wgs84(ax)

    # ------------------------------------------------------------------
    # Build panel list: (title, kind, data, cmap)
    # ------------------------------------------------------------------
    # kind is one of: "catchment" | "bng" | "aspect" | "soil" | "eo_raster"
    panels: list[tuple[str, str, Any, str]] = []

    panels.append(("Catchment", "catchment", None, ""))

    sl = cs.layers

    if sl.topography is not None:
        u = sl.topography.attrs.get("units", "m")
        panels.append((f"Elevation ({u})", "bng", sl.topography, "terrain"))

    if sl.slope is not None:
        u = sl.slope.attrs.get("units", "°")
        panels.append((f"Slope ({u})", "bng", sl.slope, "YlOrRd"))

    if sl.aspect is not None:
        u = sl.aspect.attrs.get("units", "°")
        panels.append((f"Aspect ({u})", "aspect", sl.aspect, "twilight"))

    if sl.soil_polygons is not None and not sl.soil_polygons.empty:
        panels.append(("SEARG Soil Type", "soil", sl.soil_polygons, ""))

    if sl.rainfall is not None:
        rain_mean = sl.rainfall.mean(dim="time", skipna=True)
        panels.append(("Rainfall (mm/h, mean)", "bng", rain_mean, "Blues"))

    if sl.eo_bands is not None:
        _META_VARS = {"crs", "spatial_ref", "crs_wkt"}
        _real_vars = [k for k in sl.eo_bands.data_vars if k not in _META_VARS]
        if _real_vars:
            _green_key = "B03" if "B03" in sl.eo_bands.data_vars else _real_vars[0]
            panels.append(("Green band (B03, mean)", "eo_raster", sl.eo_bands[_green_key], "YlGn"))

    if sl.ndvi is not None:
        panels.append(("NDVI (mean)", "eo_raster", sl.ndvi, "RdYlGn"))

    n = len(panels)
    if n == 0:
        logger.warning("No panels to plot — skipping layer plot.")
        return

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.5, nrows * 4.5),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    # ------------------------------------------------------------------
    # Render each panel
    # ------------------------------------------------------------------
    for idx, (title, kind, data, cmap) in enumerate(panels):
        ax = axes_flat[idx]

        # ── 1. Catchment panel (WGS-84, with flow accumulation) ──────────
        if kind == "catchment":
            # Limits and basemap first (contextily needs limits set).
            xlo, xhi, ylo, yhi = _wgs_buf()
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.set_title(title, fontsize=9, pad=4)
            ax.set_xlabel("Longitude (°)", fontsize=7)
            ax.set_ylabel("Latitude (°)", fontsize=7)
            ax.tick_params(labelsize=7)
            _add_basemap(ax, "EPSG:4326")

            # ── Flow accumulation background ─────────────────────────────
            if dem_path is not None:
                try:
                    import rasterio.features as _rf
                    import rasterio.transform as _rt

                    from eoflow.catchment import compute_flow_accumulation

                    acc_arr, acc_tf, _ = compute_flow_accumulation(dem_path)
                    H_dem, W_dem = acc_arr.shape

                    # Crop array to the catchment WGS-84 bbox (+ small pixel pad).
                    _pad = 5
                    col_s = max(0, int((bbox_wgs[0] - acc_tf.c) / acc_tf.a) - _pad)
                    col_e = min(W_dem, int((bbox_wgs[2] - acc_tf.c) / acc_tf.a) + _pad)
                    # acc_tf.e is negative (north-up), so y → row conversion uses it directly.
                    row_s = max(0, int((bbox_wgs[3] - acc_tf.f) / acc_tf.e) - _pad)
                    row_e = min(H_dem, int((bbox_wgs[1] - acc_tf.f) / acc_tf.e) + _pad)

                    acc_crop = acc_arr[row_s:row_e, col_s:col_e].copy()

                    # Geographic extent of the cropped tile.
                    cx_min = acc_tf.c + col_s * acc_tf.a
                    cx_max = acc_tf.c + col_e * acc_tf.a
                    cy_max = acc_tf.f + row_s * acc_tf.e  # north edge
                    cy_min = acc_tf.f + row_e * acc_tf.e  # south edge

                    # Log-transform (spans many orders of magnitude).
                    log_acc = np.log1p(acc_crop)

                    # Mask cells outside the catchment polygon.
                    Hc, Wc = acc_crop.shape
                    affine_crop = _rt.from_bounds(cx_min, cy_min, cx_max, cy_max, Wc, Hc)
                    outside = _rf.geometry_mask(
                        [catchment_wgs84.__geo_interface__],
                        out_shape=(Hc, Wc),
                        transform=affine_crop,
                        invert=False,
                    )
                    log_acc[outside] = np.nan

                    cmap_acc = plt.get_cmap("Blues").copy()
                    cmap_acc.set_bad(alpha=0.0)
                    im_acc = ax.imshow(
                        log_acc,
                        extent=(cx_min, cx_max, cy_min, cy_max),
                        origin="upper",
                        cmap=cmap_acc,
                        aspect="auto",
                        alpha=0.80,
                        zorder=2,
                    )
                    plt.colorbar(
                        im_acc,
                        ax=ax,
                        shrink=0.75,
                        pad=0.03,
                        fraction=0.046,
                        label="log₁₊(upstream cells)",
                    )
                except Exception as exc:
                    logger.warning("Flow accumulation display failed: %s", exc)

            # Catchment outline (no fill — flow acc or basemap is the bg).
            catchment_wgs84_gs.plot(
                ax=ax,
                facecolor="none",
                edgecolor="#1a5276",
                linewidth=1.8,
                zorder=5,
            )

            # Sample point + pour point with legend.
            _mark_points_wgs84(ax, legend=True)

            area = cs.catchment_area_km2()
            if area is not None:
                ax.text(
                    0.5,
                    0.02,
                    f"Area ≈ {area:.1f} km²",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
                    zorder=9,
                )

        # ── 2. Generic BNG raster (elevation, slope, rainfall) ──────────
        elif kind == "bng":
            _render_bng_raster(ax, data, title, cmap)
            _mark_points_bng(ax)

        # ── 3. Aspect (cyclic colourmap, fixed 0–360 range) ─────────────
        elif kind == "aspect":
            _render_bng_raster(ax, data, title, cmap, cyclic=True)
            _mark_points_bng(ax)

        # ── 4. SEARG soil-type polygons ──────────────────────────────────
        elif kind == "soil":
            soil_gdf = data

            # Clip polygons to the catchment boundary so nothing bleeds out.
            try:
                clipped = soil_gdf.copy()
                clipped["geometry"] = soil_gdf.geometry.intersection(catchment_bng)
                soil_gdf = clipped[
                    clipped.geometry.notna() & ~clipped.geometry.is_empty
                ].reset_index(drop=True)
            except Exception as exc:
                logger.debug("Soil clip failed (%s) — using unclipped polygons.", exc)

            # Set limits and basemap before drawing polygons.
            xlo, xhi, ylo, yhi = _bng_buf()
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.set_aspect("equal", adjustable="datalim")
            _style_bng_ax(ax, title)
            _add_basemap(ax, "EPSG:27700")

            present_groups = soil_gdf["SEARG_Concise"].dropna().unique()
            for group in present_groups:
                colour = SEARG_COLOURS.get(str(group), "#808080")
                subset = soil_gdf[soil_gdf["SEARG_Concise"] == group]
                subset.plot(
                    ax=ax, color=colour, edgecolor="none", linewidth=0, alpha=0.75, zorder=2
                )

            _boundary_bng(ax)
            _mark_points_bng(ax)

            # Colour-patch legend
            legend_patches = [
                mpatches.Patch(
                    color=SEARG_COLOURS.get(str(g), "#808080"),
                    label=str(g),
                )
                for g in present_groups
            ]
            ax.legend(
                handles=legend_patches,
                loc="lower right",
                fontsize=5,
                framealpha=0.85,
                title="SEARG group",
                title_fontsize=6,
                borderpad=0.5,
            )

        # ── 5. openEO raster (green band, NDVI, or any single-band DA) ───
        elif kind == "eo_raster":
            _render_eo_raster(ax, data, title, cmap)
            _mark_points_wgs84(ax)

        else:
            ax.set_visible(False)

    # Hide any unused axes in the grid.
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Overall figure title with site metadata.
    try:
        area_km2 = cs.catchment_area_km2()
        area_str = f" — area ≈ {area_km2:.1f} km²" if area_km2 is not None else ""
    except Exception:
        area_str = ""

    fig.suptitle(
        f"{cs.site_name}  ({cs.notation}){area_str}",
        fontsize=12,
        y=1.005,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info("Layer plot saved to %s", output_path)


def _plot_presentation_tile(
    cs: Sample,
    output_path: Path,
    *,
    dem_path: Path | None = None,
) -> None:
    """Create a 2x2 tiled image of flow accumulation, aspect, soil type, and NDVI.

    Designed for presentation use with clean, compact layout.

    Parameters
    ----------
    cs : Sample
        A fully populated sample (catchment + layers).
    output_path : Path
        Destination PNG file.
    dem_path : Path, optional
        Path to the DEM GeoTIFF for flow accumulation computation.
    """
    from pyproj import Transformer

    from eoflow.soil import SEARG_COLOURS

    if not cs.has_catchment:
        logger.warning("Sample has no catchment polygon — skipping presentation tile plot.")
        return

    catchment_wgs84 = cs.catchment
    assert catchment_wgs84 is not None

    # Reproject catchment to BNG
    catchment_bng_gs = gpd.GeoSeries([catchment_wgs84], crs="EPSG:4326").to_crs("EPSG:27700")  # type: ignore[call-overload]
    catchment_bng = catchment_bng_gs.iloc[0]
    catchment_wgs84_gs = gpd.GeoSeries([catchment_wgs84], crs="EPSG:4326")  # type: ignore[call-overload]

    bbox_bng = catchment_bng.bounds
    bbox_wgs = catchment_wgs84.bounds

    # Helper functions
    def _bng_buf(frac: float = 0.08):  # Smaller buffer for compact layout
        """Return (xlo, xhi, ylo, yhi) BNG limits with fractional padding."""
        dx = (bbox_bng[2] - bbox_bng[0]) * frac
        dy = (bbox_bng[3] - bbox_bng[1]) * frac
        return bbox_bng[0] - dx, bbox_bng[2] + dx, bbox_bng[1] - dy, bbox_bng[3] + dy

    def _wgs_buf(frac: float = 0.08):  # Smaller buffer for compact layout
        """Return (xlo, xhi, ylo, yhi) WGS-84 limits with fractional padding."""
        dx = (bbox_wgs[2] - bbox_wgs[0]) * frac
        dy = (bbox_wgs[3] - bbox_wgs[1]) * frac
        return bbox_wgs[0] - dx, bbox_wgs[2] + dx, bbox_wgs[1] - dy, bbox_wgs[3] + dy

    def _boundary_bng(ax: plt.Axes) -> None:  # type: ignore[name-defined]
        """Draw catchment boundary on BNG axis."""
        catchment_bng_gs.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#000000",
            linewidth=1.2,
            zorder=10,
        )

    def _boundary_wgs84(ax: plt.Axes) -> None:  # type: ignore[name-defined]
        """Draw catchment boundary on WGS-84 axis."""
        catchment_wgs84_gs.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#000000",
            linewidth=1.2,
            zorder=10,
        )

    # Create 2x2 figure with tight layout
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), squeeze=False)

    sl = cs.layers

    # ===== Panel 1: Flow Accumulation (top-left) =====
    ax1 = axes[0, 0]
    if dem_path is not None:
        try:
            import rasterio.features as _rf
            import rasterio.transform as _rt

            from eoflow.catchment import compute_flow_accumulation

            acc_arr, acc_tf, _ = compute_flow_accumulation(dem_path)
            H_dem, W_dem = acc_arr.shape

            # Crop to catchment bbox
            _pad = 5
            col_s = max(0, int((bbox_wgs[0] - acc_tf.c) / acc_tf.a) - _pad)
            col_e = min(W_dem, int((bbox_wgs[2] - acc_tf.c) / acc_tf.a) + _pad)
            row_s = max(0, int((bbox_wgs[3] - acc_tf.f) / acc_tf.e) - _pad)
            row_e = min(H_dem, int((bbox_wgs[1] - acc_tf.f) / acc_tf.e) + _pad)

            acc_crop = acc_arr[row_s:row_e, col_s:col_e].copy()

            cx_min = acc_tf.c + col_s * acc_tf.a
            cx_max = acc_tf.c + col_e * acc_tf.a
            cy_max = acc_tf.f + row_s * acc_tf.e
            cy_min = acc_tf.f + row_e * acc_tf.e

            log_acc = np.log1p(acc_crop)

            # Mask outside catchment
            Hc, Wc = acc_crop.shape
            affine_crop = _rt.from_bounds(cx_min, cy_min, cx_max, cy_max, Wc, Hc)
            outside = _rf.geometry_mask(
                [catchment_wgs84.__geo_interface__],
                out_shape=(Hc, Wc),
                transform=affine_crop,
                invert=False,
            )
            log_acc[outside] = np.nan

            xlo, xhi, ylo, yhi = _wgs_buf()
            ax1.set_xlim(xlo, xhi)
            ax1.set_ylim(ylo, yhi)

            cmap_acc = plt.get_cmap("Blues").copy()
            cmap_acc.set_bad(alpha=0.0)
            im_acc = ax1.imshow(
                log_acc,
                extent=(cx_min, cx_max, cy_min, cy_max),
                origin="upper",
                cmap=cmap_acc,
                aspect="auto",
                alpha=0.90,
                zorder=2,
            )
            cbar1 = plt.colorbar(im_acc, ax=ax1, shrink=0.85, pad=0.02)
            cbar1.set_label("Flow Accumulation\n(log scale)", fontsize=8)
            cbar1.ax.tick_params(labelsize=7)
            _boundary_wgs84(ax1)
            ax1.set_title("Flow Accumulation", fontsize=10, fontweight="bold", pad=8)
            ax1.tick_params(labelsize=7)
            ax1.set_xlabel("", fontsize=1)  # Minimize labels
            ax1.set_ylabel("", fontsize=1)
        except Exception as exc:
            logger.warning(f"Flow accumulation rendering failed: {exc}")
            ax1.text(
                0.5,
                0.5,
                "Flow Accumulation\nNot Available",
                ha="center",
                va="center",
                transform=ax1.transAxes,
                fontsize=10,
            )
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis("off")
    else:
        ax1.text(
            0.5,
            0.5,
            "Flow Accumulation\nNot Available",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=10,
        )
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis("off")

    # ===== Panel 2: Aspect (top-right) =====
    ax2 = axes[0, 1]
    if sl.aspect is not None:
        try:
            vals = sl.aspect.values.astype(float)
            x_c = sl.aspect.coords["x"].values
            y_c = sl.aspect.coords["y"].values

            extent = (
                float(x_c.min()),
                float(x_c.max()),
                float(y_c.min()),
                float(y_c.max()),
            )

            xlo, xhi, ylo, yhi = _bng_buf()
            ax2.set_xlim(xlo, xhi)
            ax2.set_ylim(ylo, yhi)

            im2 = ax2.imshow(
                vals,
                extent=extent,
                origin="lower",
                cmap="twilight",
                vmin=0.0,
                vmax=360.0,
                aspect="auto",
                interpolation="nearest",
                alpha=0.90,
                zorder=2,
            )
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
            cbar2.set_label("Aspect (°)", fontsize=8)
            cbar2.ax.tick_params(labelsize=7)
            _boundary_bng(ax2)
            ax2.set_title("Aspect", fontsize=10, fontweight="bold", pad=8)
            ax2.tick_params(labelsize=7)
            ax2.set_xlabel("", fontsize=1)
            ax2.set_ylabel("", fontsize=1)
        except Exception as exc:
            logger.warning(f"Aspect rendering failed: {exc}")
            ax2.text(
                0.5,
                0.5,
                "Aspect\nNot Available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=10,
            )
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis("off")
    else:
        ax2.text(
            0.5,
            0.5,
            "Aspect\nNot Available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=10,
        )
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")

    # ===== Panel 3: Soil Type (bottom-left) =====
    ax3 = axes[1, 0]
    if sl.soil_polygons is not None and not sl.soil_polygons.empty:
        try:
            soil_gdf = sl.soil_polygons.copy()

            # Clip to catchment
            try:
                clipped = soil_gdf.copy()
                clipped["geometry"] = soil_gdf.geometry.intersection(catchment_bng)
                soil_gdf = clipped[
                    clipped.geometry.notna() & ~clipped.geometry.is_empty
                ].reset_index(drop=True)
            except Exception:
                pass

            xlo, xhi, ylo, yhi = _bng_buf()
            ax3.set_xlim(xlo, xhi)
            ax3.set_ylim(ylo, yhi)
            ax3.set_aspect("equal", adjustable="datalim")

            present_groups = soil_gdf["SEARG_Concise"].dropna().unique()
            for group in present_groups:
                colour = SEARG_COLOURS.get(str(group), "#808080")
                subset = soil_gdf[soil_gdf["SEARG_Concise"] == group]
                subset.plot(
                    ax=ax3, color=colour, edgecolor="none", linewidth=0, alpha=0.85, zorder=2
                )

            _boundary_bng(ax3)

            # Compact legend
            legend_patches = [
                mpatches.Patch(
                    color=SEARG_COLOURS.get(str(g), "#808080"),
                    label=str(g)[:20],  # Truncate long labels
                )
                for g in present_groups
            ]
            ax3.legend(
                handles=legend_patches,
                loc="lower right",
                fontsize=6,
                framealpha=0.90,
                title="Soil Type",
                title_fontsize=7,
                borderpad=0.4,
            )
            ax3.set_title("Soil Type", fontsize=10, fontweight="bold", pad=8)
            ax3.tick_params(labelsize=7)
            ax3.set_xlabel("", fontsize=1)
            ax3.set_ylabel("", fontsize=1)
        except Exception as exc:
            logger.warning(f"Soil type rendering failed: {exc}")
            ax3.text(
                0.5,
                0.5,
                "Soil Type\nNot Available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=10,
            )
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis("off")
    else:
        ax3.text(
            0.5,
            0.5,
            "Soil Type\nNot Available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=10,
        )
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis("off")

    # ===== Panel 4: NDVI (bottom-right) =====
    ax4 = axes[1, 1]
    if sl.ndvi is not None:
        try:
            from pyproj import Transformer

            da = sl.ndvi

            # Handle object dtype
            if not np.issubdtype(da.dtype, np.floating):
                raw = da.values
                if raw.dtype.kind == "O":

                    def _to_f_eo(v: object) -> float:
                        if v in (b"", b"nan", "", None):
                            return np.nan
                        try:
                            return float(np.float32(v))  # type: ignore[arg-type]
                        except (ValueError, TypeError):
                            return np.nan

                    da = da.copy(data=np.vectorize(_to_f_eo)(raw).astype(np.float32))

            # Drop 'variable' dimension
            if "variable" in da.dims:
                _META = {"crs", "spatial_ref", "crs_wkt"}
                if "variable" in da.coords:
                    data_slices = [
                        str(v) for v in da.coords["variable"].values if str(v) not in _META
                    ]
                    da = (
                        da.sel(variable=data_slices[0], drop=True)
                        if data_slices
                        else da.isel(variable=0, drop=True)
                    )
                else:
                    da = da.isel(variable=0, drop=True)

            # Collapse time
            for time_dim in ("t", "time"):
                if time_dim in da.dims:
                    da = da.mean(dim=time_dim, skipna=True)
                    break
            da = da.squeeze(drop=True)

            if da.ndim == 2:
                # Get coordinates
                y_c = np.linspace(0.0, 1.0, da.shape[0])
                for yn in ("latitude", "lat", "y"):
                    if yn in da.coords:
                        y_c = da.coords[yn].values
                        break
                x_c = np.linspace(0.0, 1.0, da.shape[1])
                for xn in ("longitude", "lon", "x"):
                    if xn in da.coords:
                        x_c = da.coords[xn].values
                        break

                x_min_v = float(x_c.min())
                x_max_v = float(x_c.max())
                y_min_v = float(y_c.min())
                y_max_v = float(y_c.max())

                # Determine CRS and WGS-84 extent
                if -90.0 <= y_min_v and y_max_v <= 90.0:
                    wgs_x_min, wgs_x_max = x_min_v, x_max_v
                    wgs_y_min, wgs_y_max = y_min_v, y_max_v
                elif (
                    0.0 <= y_min_v
                    and y_max_v <= 800_000.0
                    and 0.0 <= x_min_v
                    and x_max_v <= 800_000.0
                ):
                    tr_b = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                    c_lon, c_lat = tr_b.transform(
                        [x_min_v, x_max_v, x_min_v, x_max_v],
                        [y_min_v, y_min_v, y_max_v, y_max_v],
                    )
                    wgs_x_min, wgs_x_max = float(min(c_lon)), float(max(c_lon))
                    wgs_y_min, wgs_y_max = float(min(c_lat)), float(max(c_lat))
                else:
                    _clon = catchment_wgs84.centroid.x
                    _clat = catchment_wgs84.centroid.y
                    _zone = int((_clon + 180.0) / 6.0) + 1
                    _epsg = 32600 + _zone if _clat >= 0.0 else 32700 + _zone
                    tr_u = Transformer.from_crs(f"EPSG:{_epsg}", "EPSG:4326", always_xy=True)
                    c_lon, c_lat = tr_u.transform(
                        [x_min_v, x_max_v, x_min_v, x_max_v],
                        [y_min_v, y_min_v, y_max_v, y_max_v],
                    )
                    wgs_x_min, wgs_x_max = float(min(c_lon)), float(max(c_lon))
                    wgs_y_min, wgs_y_max = float(min(c_lat)), float(max(c_lat))

                # Mask outside catchment
                vals = da.values.astype(float)
                try:
                    import rasterio.features
                    import rasterio.transform

                    H, W = vals.shape
                    affine = rasterio.transform.from_bounds(
                        wgs_x_min, wgs_y_min, wgs_x_max, wgs_y_max, W, H
                    )
                    outside = rasterio.features.geometry_mask(
                        [catchment_wgs84.__geo_interface__],
                        out_shape=(H, W),
                        transform=affine,
                        invert=False,
                    )
                    vals[outside] = np.nan
                except Exception:
                    pass

                # Flip if needed
                if len(y_c) > 1 and float(y_c[0]) > float(y_c[-1]):
                    vals = vals[::-1, :]

                # Auto-scale
                finite = vals[np.isfinite(vals)]
                vmin_use, vmax_use = -1.0, 1.0
                if finite.size > 0:
                    vmin_use = float(np.percentile(finite, 2))
                    vmax_use = float(np.percentile(finite, 98))

                wgs_extent = (wgs_x_min, wgs_x_max, wgs_y_min, wgs_y_max)

                xlo, xhi, ylo, yhi = _wgs_buf()
                ax4.set_xlim(xlo, xhi)
                ax4.set_ylim(ylo, yhi)

                cmap_masked = plt.get_cmap("RdYlGn").copy()
                cmap_masked.set_bad(alpha=0.0)

                im4 = ax4.imshow(
                    vals,
                    extent=wgs_extent,
                    origin="lower",
                    cmap=cmap_masked,
                    vmin=vmin_use,
                    vmax=vmax_use,
                    aspect="auto",
                    interpolation="nearest",
                    alpha=0.90,
                    zorder=2,
                )
                cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.85, pad=0.02)
                cbar4.set_label("NDVI", fontsize=8)
                cbar4.ax.tick_params(labelsize=7)
                _boundary_wgs84(ax4)
                ax4.set_title("NDVI", fontsize=10, fontweight="bold", pad=8)
                ax4.tick_params(labelsize=7)
                ax4.set_xlabel("", fontsize=1)
                ax4.set_ylabel("", fontsize=1)
            else:
                raise ValueError("NDVI data not 2D after processing")
        except Exception as exc:
            logger.warning(f"NDVI rendering failed: {exc}")
            ax4.text(
                0.5,
                0.5,
                "NDVI\nNot Available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=10,
            )
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis("off")
    else:
        ax4.text(
            0.5,
            0.5,
            "NDVI\nNot Available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=10,
        )
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis("off")

    # Overall figure title
    try:
        area_km2 = cs.catchment_area_km2()
        area_str = f" — Area: {area_km2:.1f} km²" if area_km2 is not None else ""
    except Exception:
        area_str = ""

    fig.suptitle(
        f"{cs.site_name} ({cs.notation}){area_str}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Presentation tile plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def main():
    # 1.1 Load water quality sample data to Dataframe
    wq_sample_path = PROJECT_ROOT / "data/wq_samples/devon_water_quality_10.csv"
    assert wq_sample_path.exists() and wq_sample_path.suffix == ".csv"
    logger.info(f"Loading water quality sample data from {wq_sample_path}")

    wq_sample_df = pd.read_csv(wq_sample_path)
    assert wq_sample_df.shape[0] > 1
    logger.info(f"Loaded {wq_sample_df.shape[0]} water quality samples")

    # 1.2 Load Devon DEM file
    devon_dem_path = DATA_DIR / "dems/devon_dem_cop30.tif"
    assert devon_dem_path.exists() and devon_dem_path.suffix == ".tif"
    logger.info(f"Loaded Devon DEM from {devon_dem_path}")

    # 2. Create single Sample
    # 2.1 Select single row from df and convert to Sample
    SAMPLE_IND = 0
    row_series = wq_sample_df.iloc[SAMPLE_IND]

    cs = Sample(
        row=row_series,
        catchment=None,
    )
    logger.info(f"Created Sample from row {SAMPLE_IND}")

    # Setup checkpointing tools
    checkpoint_path = DATA_DIR / "temp/sample_flow_checkpoint_{:03d}"

    def checkpoint_step(s: int) -> Path:
        return Path(str(checkpoint_path).format(s))

    # 3. Catchment Delineation
    # 3.1 Delineate catchment from DEM
    chk = checkpoint_step(1)
    if chk.exists() and chk.is_dir():
        # load from checkpoint if it exists
        cs = Sample.load(chk)
        logger.info(f"Loaded catchment from checkpoint {chk}")
    else:
        # otherwise, delineate from DEM and save checkpoint
        FLOW_ACC_THRESH = 5000
        assert cs.catchment is None
        cs.delineate(dem_path=devon_dem_path, flow_acc_threshold=FLOW_ACC_THRESH)
        assert cs.catchment is not None
        logger.info(f"Delineated catchment from DEM for sample {cs.id}")
        cs.save(chk)
        logger.info(f"Saved sample data to checkpoint {chk}")

    # 4. Compute layers
    # 4.1 Compute topography & slope
    START_DATETIME = datetime.strptime("2024-03-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
    END_DATETIME = datetime.strptime("2024-03-01T01:00:00", "%Y-%m-%dT%H:%M:%S")
    RAINFALL_DOWNLOAD_PATH = Path(PROJECT_ROOT / "data/temp/rainfall_001")
    chk = checkpoint_step(2)
    if chk.exists() and chk.is_dir():
        # load from checkpoint if it exists
        cs = Sample.load(chk)
        logger.info(f"Loaded catchment from checkpoint {chk}")
    else:
        cs.compute_layers(
            dem_path=DATA_DIR / "dems/devon_dem_cop30.tif",
            rainfall_start=START_DATETIME,
            rainfall_end=END_DATETIME,
            rainfall_download_dir=RAINFALL_DOWNLOAD_PATH,
        )
        logger.info(f"Computed layers for sample {cs.id}")
        cs.save(chk)
        logger.info(f"Saved sample data to checkpoint {chk}")

    # 5. EO Queries (NDVI + NDWI)
    # Fetch NDVI and NDWI for the catchment using openEO / Sentinel-2.
    # A separate checkpoint is used so the potentially slow download is not
    # repeated on re-runs.
    EO_START_DATE = "2024-02-01"
    EO_END_DATE = "2024-04-30"
    chk = checkpoint_step(3)
    if chk.exists() and chk.is_dir():
        cs = Sample.load(chk)
        logger.info(f"Loaded EO data from checkpoint {chk}")
    else:
        try:
            logger.info(
                f"Connecting to openEO to fetch NDVI and NDWI ({EO_START_DATE} → {EO_END_DATE}) …"
            )
            conn = Sample.connect_openeo()

            cs.fetch_ndvi(conn, start_date=EO_START_DATE, end_date=EO_END_DATE)
            logger.info(f"NDVI fetched for sample {cs.id}")

            cs.fetch_ndwi(conn, start_date=EO_START_DATE, end_date=EO_END_DATE)
            logger.info(f"NDWI fetched for sample {cs.id}")

            cs.save(chk)
            logger.info(f"Saved EO data to checkpoint {chk}")
        except Exception as exc:
            logger.warning(f"EO query step skipped — could not fetch NDVI/NDWI: {exc}")

    # 5a. Green band (B03)
    # Fetch the raw Sentinel-2 green reflectance band so it can be displayed
    # alongside the derived indices.  Stored in layers.eo_bands.
    # A separate checkpoint (4) avoids re-downloading on repeated runs.
    chk = checkpoint_step(4)
    if chk.exists() and chk.is_dir():
        cs = Sample.load(chk)
        logger.info(f"Loaded green band from checkpoint {chk}")
    else:
        try:
            logger.info(
                f"Connecting to openEO to fetch green band B03 ({EO_START_DATE} → {EO_END_DATE}) …"
            )
            conn = Sample.connect_openeo()
            cs.fetch_bands(conn, bands=["B03"], start_date=EO_START_DATE, end_date=EO_END_DATE)
            logger.info(f"Green band (B03) fetched for sample {cs.id}")
            cs.save(chk)
            logger.info(f"Saved green band to checkpoint {chk}")
        except Exception as exc:
            logger.warning(f"Green band fetch skipped — could not fetch B03: {exc}")

    # 6. Visualize
    # 6.1 Load county boundary
    COUNTY_BOUNDARY_DIR = DATA_DIR / "shapefiles/devon_county"
    assert COUNTY_BOUNDARY_DIR.exists() and COUNTY_BOUNDARY_DIR.is_dir()
    county_boundary = gpd.read_file(COUNTY_BOUNDARY_DIR.glob("*.shp").__next__())

    # 6.2 Generate base folium map with county boundary
    m = make_base_map_from_poly(county_boundary)

    # 6.3 Add county boundary
    add_area(county_boundary, name="County Boundary", f_map=m)

    # 6.4 Add sample catchment boundary
    add_sample(cs, f_map=m, catchment_color="green", raster_layers="auto")

    # 6.4b Flow accumulation overlay
    # Compute log-scaled flow accumulation from the DEM, crop to the catchment
    # bounding box, and add as a folium ImageOverlay using the same
    # _make_index_overlay helper that handles NDVI/NDWI.
    try:
        import xarray as xr

        from eoflow.catchment import compute_flow_accumulation
        from eoflow.folium import _make_index_overlay

        logger.info("Computing flow accumulation for folium overlay …")
        acc_arr, acc_tf, _ = compute_flow_accumulation(devon_dem_path)
        H_dem, W_dem = acc_arr.shape

        # Crop to catchment bbox + a small buffer so the streams near the
        # boundary are fully visible.
        _catchment = cs.catchment
        assert _catchment is not None  # guaranteed by has_catchment check at step 3
        _bbox = _catchment.bounds  # (W, S, E, N)
        _buf = 0.02  # degrees
        col_s = max(0, int((_bbox[0] - _buf - acc_tf.c) / acc_tf.a))
        col_e = min(W_dem, int((_bbox[2] + _buf - acc_tf.c) / acc_tf.a) + 1)
        # acc_tf.e is negative (north-up raster), so larger row index → more south.
        row_s = max(0, int((_bbox[3] + _buf - acc_tf.f) / acc_tf.e))
        row_e = min(H_dem, int((_bbox[1] - _buf - acc_tf.f) / acc_tf.e) + 1)

        acc_crop = acc_arr[row_s:row_e, col_s:col_e]
        log_acc = np.log1p(acc_crop.astype(float))

        # Build pixel-centre coordinate arrays (y descending = north-first).
        lons = np.linspace(
            acc_tf.c + (col_s + 0.5) * acc_tf.a,
            acc_tf.c + (col_e - 0.5) * acc_tf.a,
            col_e - col_s,
        )
        lats = np.linspace(
            acc_tf.f + (row_s + 0.5) * acc_tf.e,
            acc_tf.f + (row_e - 0.5) * acc_tf.e,
            row_e - row_s,
        )  # descending: lats[0] is northernmost row

        da_acc = xr.DataArray(
            log_acc,
            dims=["y", "x"],
            coords={"y": lats, "x": lons},
            name="flow_accumulation",
        )

        from shapely.geometry import MultiPolygon, Polygon

        _catchment_geom = _catchment if isinstance(_catchment, (Polygon, MultiPolygon)) else None
        acc_overlay = _make_index_overlay(
            da_acc,
            cmap="Blues",
            label="Flow Accumulation (log scale)",
            vmin=0.0,
            vmax=float(np.nanmax(log_acc)) if np.any(np.isfinite(log_acc)) else 1.0,
            catchment=_catchment_geom,
            fallback_bbox=cs.catchment_bbox(),
        )
        if acc_overlay is not None:
            acc_overlay.add_to(m)
            logger.info("Flow accumulation overlay added to folium map.")
    except Exception as exc:
        logger.warning(f"Could not add flow accumulation overlay to folium map: {exc}")

    # 6.5 Finalise the layer control once all layers are present, then render.
    folium.LayerControl(collapsed=False).add_to(m)

    # 6.6 Save visualisation to HTML and open in browser (non-blocking)
    output_path = PROJECT_ROOT / "outputs/sample_flow_map.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info(f"Map saved to {output_path}")
    webbrowser.open(output_path.as_uri())

    # 7. Static matplotlib layer overview
    # Produce a single PNG that shows the catchment polygon followed by each
    # computed spatial layer (elevation, slope, aspect, soil, rainfall, NDVI),
    # with the catchment boundary drawn over each panel and a colourbar /
    # legend where appropriate.
    layer_plot_path = PROJECT_ROOT / "outputs/sample_flow_layers.png"
    _plot_sample_layers(cs, layer_plot_path, dem_path=DATA_DIR / "dems/devon_dem_cop30.tif")

    # 8. Presentation tile (2x2 compact layout)
    # Produce a clean 2x2 tiled image with flow accumulation, aspect, soil type,
    # and NDVI for use in presentations.
    presentation_tile_path = PROJECT_ROOT / "outputs/sample_flow_presentation_tile.png"
    _plot_presentation_tile(
        cs, presentation_tile_path, dem_path=DATA_DIR / "dems/devon_dem_cop30.tif"
    )

    logger.info(f"script {__file__} finished without error!")


if __name__ == "__main__":
    main()
