"""
Visualise a DEM GeoTIFF as a colour-coded elevation overlay on an
interactive map.

Reads a GeoTIFF produced by ``get_dem.py`` (or any EPSG:4326 DEM) and
displays it as a semi-transparent colour-coded raster overlay on an
OpenStreetMap base layer.  The resulting HTML page is written to disk
and opened automatically in the default web browser.

Usage
-----
    python -m scripts.visualise_dem \
        --input data/uk_dem.geotiff

    # Optionally control output path, opacity, and max display resolution
    python -m scripts.visualise_dem \
        --input data/uk_dem.geotiff \
        --output dem_map.html \
        --opacity 0.7 \
        --max-pixels 2048
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import tempfile
import warnings
import webbrowser
from pathlib import Path
from typing import cast

import branca.colormap as cm
import folium
import folium.raster_layers  # noqa: F401 – ensure submodule is importable
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from eoflow.log_utils import get_logger

logger = get_logger(__file__)


# ---------------------------------------------------------------------------
# Colour-map helpers
# ---------------------------------------------------------------------------


def _make_terrain_cmap() -> mcolors.LinearSegmentedColormap:
    """Build a perceptually-pleasant terrain colourmap.

    Low elevations are green, mid-range brown/tan, and high elevations
    transition to white (snow-capped peaks).  This is close to the
    classic *wiki-schwarzwald* style used in cartography.
    """
    colours = [
        (0.00, "#1a6e2e"),  # deep green  (lowlands)
        (0.15, "#5fa843"),  # green
        (0.30, "#a1c34a"),  # yellow-green
        (0.45, "#ddd07a"),  # tan
        (0.60, "#c4944a"),  # brown
        (0.75, "#9a7250"),  # dark brown  (mountains)
        (0.90, "#d4c8b8"),  # light grey  (rock)
        (1.00, "#f5f5f5"),  # near-white  (snow)
    ]
    positions = [c[0] for c in colours]
    hex_colours = [c[1] for c in colours]
    rgb = [mcolors.to_rgb(h) for h in hex_colours]
    return mcolors.LinearSegmentedColormap.from_list("terrain_eoflow", list(zip(positions, rgb)))


def _compute_hillshade(
    elevation: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    """Compute a simple hillshade from an elevation array.

    Returns values in [0, 1] where 1 is fully illuminated.
    """
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    # Gradient (Sobel-like via np.gradient)
    dy, dx = np.gradient(elevation.astype(float))

    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)

    shade = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(
        az_rad - aspect
    )
    return np.clip(shade, 0, 1)


# ---------------------------------------------------------------------------
# Raster reading with reprojection to Web Mercator
# ---------------------------------------------------------------------------

# Transformer to convert EPSG:3857 bounds back to EPSG:4326 for folium
_MERCATOR_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def _read_dem(
    path: Path,
    max_pixels: int = 2048,
) -> tuple[np.ndarray, float, BoundingBox]:
    """Read a DEM GeoTIFF and reproject to Web Mercator for display.

    Leaflet/folium renders tiles in Web Mercator (EPSG:3857) and
    stretches ``ImageOverlay`` images *linearly* in screen-space between
    the corner coordinates.  If the source image has pixels that are
    evenly spaced in geographic degrees (EPSG:4326), intermediate
    latitudes will not line up with the basemap.  Reprojecting to
    EPSG:3857 first ensures the pixel grid matches Leaflet's projection.

    Parameters
    ----------
    path : Path
        Path to the GeoTIFF.
    max_pixels : int
        Maximum number of pixels on the longest axis.  Rasters larger
        than this are down-sampled for display.

    Returns
    -------
    data : np.ndarray
        2-D elevation array (float32, nodata replaced with NaN) in
        Web Mercator pixel space.
    nodata : float
        Original nodata value from the file.
    bounds : rasterio.coords.BoundingBox
        Geographic bounding box (EPSG:4326) of the reprojected raster,
        suitable for passing to folium's ``ImageOverlay``.
    """
    with rasterio.open(path) as src:
        nodata_val = src.nodata if src.nodata is not None else -32768.0
        src_crs = src.crs or "EPSG:4326"

        # --- Compute the reprojection transform to EPSG:3857 -------------
        dst_crs = "EPSG:3857"
        transform_3857, w_3857, h_3857 = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )
        width_3857: int = int(cast(int, w_3857))
        height_3857: int = int(cast(int, h_3857))

        # --- Optionally cap resolution to max_pixels ----------------------
        longest = max(height_3857, width_3857)
        if longest > max_pixels:
            scale = max_pixels / longest
            width_3857 = max(1, int(width_3857 * scale))
            height_3857 = max(1, int(height_3857 * scale))
            # Recalculate transform for the new dimensions
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
            logger.info(
                "Reprojecting + down-sampling to %d×%d for display",
                height_3857,
                width_3857,
            )

        # --- Reproject into the Mercator array ----------------------------
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

    # Replace nodata with NaN
    dst_array[np.isclose(dst_array, nodata_val)] = np.nan

    # --- Compute EPSG:4326 bounds of the reprojected raster ---------------
    # The Mercator bounds in meters:
    merc_left = transform_3857.c
    merc_top = transform_3857.f
    merc_right = merc_left + width_3857 * transform_3857.a
    merc_bottom = merc_top + height_3857 * transform_3857.e  # e is negative

    # Convert corners to lon/lat
    lon_min, lat_min = _MERCATOR_TO_WGS84.transform(merc_left, merc_bottom)
    lon_max, lat_max = _MERCATOR_TO_WGS84.transform(merc_right, merc_top)

    geo_bounds = BoundingBox(left=lon_min, bottom=lat_min, right=lon_max, top=lat_max)

    logger.info(
        "Reprojected bounds (WGS 84): S=%.4f W=%.4f N=%.4f E=%.4f",
        geo_bounds.bottom,
        geo_bounds.left,
        geo_bounds.top,
        geo_bounds.right,
    )

    return dst_array, nodata_val, geo_bounds


# ---------------------------------------------------------------------------
# Render elevation array → PNG bytes
# ---------------------------------------------------------------------------


def _render_png(
    elevation: np.ndarray,
    vmin: float,
    vmax: float,
    hillshade: bool = True,
) -> bytes:
    """Render an elevation array as an RGBA PNG byte string.

    NaN pixels are fully transparent.

    Parameters
    ----------
    elevation : np.ndarray
        2-D float array with NaN for nodata.
    vmin, vmax : float
        Colour-scale limits.
    hillshade : bool
        Whether to modulate colours with a hillshade effect.

    Returns
    -------
    bytes
        PNG-encoded image.
    """
    cmap = _make_terrain_cmap()

    # Normalise to [0, 1]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    normed = norm(elevation)

    # Apply colourmap → RGBA float array [0, 1]
    rgba = cmap(normed)  # shape (H, W, 4)

    # Apply hillshade
    if hillshade:
        shade = _compute_hillshade(elevation)
        # Blend: darken by shade, keep some ambient light
        ambient = 0.3
        factor = ambient + (1.0 - ambient) * shade
        rgba[..., :3] *= factor[..., np.newaxis]
        rgba[..., :3] = np.clip(rgba[..., :3], 0, 1)

    # Make nodata transparent
    nan_mask = np.isnan(elevation)
    rgba[nan_mask, 3] = 0.0

    # Convert to uint8 (NaN pixels produce a harmless "invalid value" warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rgba_u8 = (rgba * 255).astype(np.uint8)

    # Encode as PNG
    buf = io.BytesIO()
    plt.imsave(buf, rgba_u8, format="png")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Core visualisation
# ---------------------------------------------------------------------------


def visualise(
    input_path: Path,
    output_html: Path | None = None,
    *,
    opacity: float = 0.65,
    max_pixels: int = 2048,
    hillshade: bool = True,
) -> Path:
    """Build an interactive map with a DEM overlay and open it in the browser.

    Parameters
    ----------
    input_path : Path
        Path to the DEM GeoTIFF file.
    output_html : Path or None
        Where to write the HTML file.  A temporary file is used when *None*.
    opacity : float
        Base opacity of the DEM overlay (0 = invisible, 1 = opaque).
    max_pixels : int
        Maximum pixels on the longest raster axis before down-sampling.
    hillshade : bool
        Whether to apply a hillshade effect to the elevation rendering.

    Returns
    -------
    Path
        The path to the written HTML file.
    """
    # --- Read raster ------------------------------------------------------
    elevation, _nodata, bounds = _read_dem(input_path, max_pixels=max_pixels)

    valid = elevation[~np.isnan(elevation)]
    if valid.size == 0:
        raise ValueError("DEM contains no valid (non-nodata) pixels.")

    vmin = float(np.nanmin(valid))
    vmax = float(np.nanmax(valid))
    logger.info(
        "Elevation range: %d – %d m   (%d×%d display pixels)",
        vmin,
        vmax,
        elevation.shape[0],
        elevation.shape[1],
    )

    # --- Render to PNG ----------------------------------------------------
    png_bytes = _render_png(elevation, vmin, vmax, hillshade=hillshade)
    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    data_uri = f"data:image/png;base64,{png_b64}"

    logger.info("Rendered PNG overlay: %d KB", len(png_bytes) // 1024)

    # --- Build folium map -------------------------------------------------
    centre_lat = (bounds.bottom + bounds.top) / 2
    centre_lon = (bounds.left + bounds.right) / 2

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=7,
        tiles="OpenStreetMap",
    )

    # Image overlay with the rendered DEM
    img_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

    folium.raster_layers.ImageOverlay(
        image=data_uri,
        bounds=img_bounds,
        opacity=opacity,
        name="DEM Elevation",
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # Layer control so the user can toggle the overlay
    folium.LayerControl(collapsed=False).add_to(m)

    # Colour legend
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
    colormap = cm.LinearColormap(
        colors=terrain_colours,
        vmin=vmin,
        vmax=vmax,
        caption="Elevation (m)",
    )
    colormap.add_to(m)

    # Fit map to the raster extent
    sw = [bounds.bottom, bounds.left]
    ne = [bounds.top, bounds.right]
    m.fit_bounds([sw, ne], padding=[20, 20])

    # --- Inject a simple opacity slider via custom HTML -------------------
    slider_html = f"""
    <div style="
        position: fixed; bottom: 50px; left: 10px; z-index: 9999;
        background: white; padding: 10px 14px; border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: sans-serif;
        font-size: 13px;
    ">
        <label for="dem-opacity" style="margin-right: 6px;">DEM opacity</label>
        <input type="range" id="dem-opacity" min="0" max="100"
               value="{int(opacity * 100)}" style="width: 120px; vertical-align: middle;">
        <span id="dem-opacity-val" style="margin-left: 4px;">{int(opacity * 100)}%</span>
    </div>
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        var slider = document.getElementById('dem-opacity');
        var label  = document.getElementById('dem-opacity-val');
        // Find the overlay image element(s)
        function setOpacity(val) {{
            var imgs = document.querySelectorAll('.leaflet-image-layer');
            imgs.forEach(function(img) {{ img.style.opacity = val; }});
        }}
        slider.addEventListener('input', function() {{
            var v = this.value / 100;
            label.textContent = this.value + '%';
            setOpacity(v);
        }});
    }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(slider_html))  # type: ignore[attr-defined]

    # --- Write HTML and open in browser -----------------------------------
    if output_html is not None:
        html_path = Path(output_html).resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", prefix="dem_vis_", delete=False)
        tmp.close()
        html_path = Path(tmp.name).resolve()

    m.save(str(html_path))
    print(f"Map saved to {html_path}  ({elevation.shape[0]}×{elevation.shape[1]} pixels)")
    webbrowser.open(html_path.as_uri())
    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualise a DEM GeoTIFF as a colour-coded elevation overlay "
            "on an interactive map opened in the browser."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the DEM GeoTIFF file.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the HTML map.  A temporary file is used when omitted.",
    )
    p.add_argument(
        "--opacity",
        type=float,
        default=0.65,
        help="Initial opacity of the DEM overlay (0.0 – 1.0).",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=2048,
        help=(
            "Maximum pixels on the longest raster axis.  Larger rasters "
            "are down-sampled for display.  Increase for sharper detail at "
            "the cost of a larger HTML file."
        ),
    )
    p.add_argument(
        "--no-hillshade",
        action="store_true",
        default=False,
        help="Disable the hillshade effect (flat colour only).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not (0.0 <= args.opacity <= 1.0):
        print(f"Error: --opacity must be between 0.0 and 1.0, got {args.opacity}", file=sys.stderr)
        sys.exit(1)

    try:
        visualise(
            args.input,
            output_html=args.output,
            opacity=args.opacity,
            max_pixels=args.max_pixels,
            hillshade=not args.no_hillshade,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
