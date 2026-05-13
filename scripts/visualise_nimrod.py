"""
Visualise NIMROD rainfall data on an interactive map.

Reads NetCDF files produced by ``process_nimrod_local.py`` or ``download_nimrod.py``
and displays the rainfall data as a semi-transparent raster overlay on a
Leaflet map, with the region boundary shown.

The resulting HTML page is written and opened automatically in the default
web browser.

Usage
-----
::

    # Visualize a single NetCDF file
    python -m scripts.visualise_nimrod \
        --input data/nimrod_processed/2026/20260101/20260101_120000.nc \
        --shapefile data/shapefiles/devon_county \
        --output data/nimrod_map.html

    # Visualize the mean of a date's worth of data
    python -m scripts.visualise_nimrod \
        --input data/nimrod_processed/2026/20260101 \
        --shapefile data/shapefiles/devon_county \
        --aggregate mean \
        --output data/nimrod_mean.html

    # Visualize the max rainfall for a date
    python -m scripts.visualise_nimrod \
        --input data/nimrod_processed/2026/20260101 \
        --aggregate max

    # Visualize the mean of all data in a directory tree (recursive)
    python -m scripts.visualise_nimrod \
        --input data/nimrod_processed/2026 \
        --shapefile data/shapefiles/devon_county \
        --aggregate mean \
        --recursive \
        --output data/nimrod_2026_mean.html

    # Compute sum without opening browser
    python -m scripts.visualise_nimrod \
        --input data/nimrod_processed/2026 \
        --shapefile data/shapefiles/devon_county \
        --aggregate sum \
        --no-open \
        --output data/nimrod_sum.html
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
import webbrowser
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import iris
import numpy as np

from eoflow.log_utils import get_logger

# Suppress Iris metadata warnings about non-contiguous bounds
warnings.filterwarnings("ignore", category=iris.warnings.IrisVagueMetadataWarning)

logger = get_logger(__name__)


def load_shapefile(shapefile_path: Path) -> gpd.GeoDataFrame:
    """Load shapefile and reproject to WGS84 (EPSG:4326) for Leaflet.

    Parameters
    ----------
    shapefile_path :
        Path to shapefile or directory containing shapefile.

    Returns
    -------
    gpd.GeoDataFrame
        Shapefile in WGS84.
    """
    if shapefile_path.is_dir():
        # Find .shp file in directory
        shp_files = list(shapefile_path.glob("*.shp"))
        if not shp_files:
            raise ValueError(f"No .shp file found in {shapefile_path}")
        shapefile_path = shp_files[0]

    gdf = gpd.read_file(shapefile_path)

    # Reproject to WGS84 if needed
    if gdf.crs is None:
        logger.warning("Shapefile has no CRS, assuming EPSG:27700 (BNG)")
        gdf = gdf.set_crs("EPSG:27700")

    if gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def load_netcdf(nc_path: Path) -> iris.cube.Cube:
    """Load a single NetCDF file as an Iris cube.

    Parameters
    ----------
    nc_path :
        Path to NetCDF file.

    Returns
    -------
    iris.cube.Cube
        Loaded cube.
    """
    cubes = iris.load(str(nc_path))
    if not cubes:
        raise ValueError(f"No cubes found in {nc_path}")

    # Return the first cube (should be rainfall rate)
    return cubes[0]


def load_and_aggregate_netcdf(
    input_path: Path, aggregate: str = "mean", recursive: bool = False
) -> iris.cube.Cube:
    """Load NetCDF file(s) and optionally aggregate.

    Parameters
    ----------
    input_path :
        Path to NetCDF file or directory containing NetCDF files.
    aggregate :
        Aggregation method: "mean", "max", "min", or "sum".
    recursive :
        If True and input_path is a directory, search recursively for NetCDF files.

    Returns
    -------
    iris.cube.Cube
        Loaded (and possibly aggregated) cube.
    """
    if input_path.is_file():
        return load_netcdf(input_path)

    # Load all NetCDF files in directory
    if recursive:
        nc_files = sorted(input_path.rglob("*.nc"))
    else:
        nc_files = sorted(input_path.glob("*.nc"))

    if not nc_files:
        search_type = "recursively" if recursive else "in directory"
        raise ValueError(f"No NetCDF files found {search_type} {input_path}")

    logger.info(f"Loading {len(nc_files)} NetCDF files from {input_path}")

    cubes = []
    for nc_file in nc_files:
        try:
            cube = load_netcdf(nc_file)
            cubes.append(cube)
        except Exception as e:
            logger.warning(f"Failed to load {nc_file}: {e}")

    if not cubes:
        raise ValueError("No cubes could be loaded")

    # Concatenate along time dimension
    try:
        cube_list = iris.cube.CubeList(cubes)
        merged = cube_list.merge_cube()
    except Exception:
        # If merge fails, try concatenate
        merged = iris.cube.CubeList(cubes).concatenate_cube()

    # Aggregate if requested
    if aggregate == "mean":
        logger.info("Computing mean across time")
        aggregated = merged.collapsed("time", iris.analysis.MEAN)
    elif aggregate == "max":
        logger.info("Computing max across time")
        aggregated = merged.collapsed("time", iris.analysis.MAX)
    elif aggregate == "min":
        logger.info("Computing min across time")
        aggregated = merged.collapsed("time", iris.analysis.MIN)
    elif aggregate == "sum":
        logger.info("Computing sum across time")
        aggregated = merged.collapsed("time", iris.analysis.SUM)
    else:
        raise ValueError(f"Unknown aggregate method: {aggregate}")

    return aggregated


def cube_to_leaflet_overlay(
    cube: iris.cube.Cube, colormap: cm.LinearColormap, vmin: float, vmax: float
) -> dict:
    """Convert Iris cube to data for Leaflet ImageOverlay.

    Parameters
    ----------
    cube :
        Iris cube with rainfall data in BNG coordinates.
    colormap :
        Branca colormap to apply.
    vmin :
        Minimum value for colormap range.
    vmax :
        Maximum value for colormap range.

    Returns
    -------
    dict
        Dictionary with 'image_url' and 'bounds' for Leaflet.
    """
    # Get coordinates
    y_coord = cube.coord("projection_y_coordinate")
    x_coord = cube.coord("projection_x_coordinate")

    # Get data array (ensure it's 2D)
    data = cube.data

    # Convert masked array to regular array with NaN for masked values
    if np.ma.is_masked(data):
        data = np.ma.filled(data, np.nan)

    if data.ndim > 2:
        # Squeeze out singleton dimensions
        data = np.squeeze(data)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D data after squeeze, got shape {data.shape}")

    # Convert BNG bounds to lat/lon
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

    # Get bounds in BNG (x, y)
    x_min = x_coord.points.min()
    x_max = x_coord.points.max()
    y_min = y_coord.points.min()
    y_max = y_coord.points.max()

    # Transform corners to lat/lon
    sw_lon, sw_lat = transformer.transform(x_min, y_min)
    ne_lon, ne_lat = transformer.transform(x_max, y_max)

    bounds = [[sw_lat, sw_lon], [ne_lat, ne_lon]]

    # Use the provided vmin/vmax (should match the colormap's range)
    # Validate that vmin < vmax
    if vmin >= vmax:
        raise ValueError(f"Invalid data range: vmin={vmin}, vmax={vmax}")

    # Create RGBA image
    # Flip vertically because image coordinates go top-to-bottom
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                rgba[i, j] = [0, 0, 0, 0]  # Transparent for NaN
            else:
                # Clamp value to valid range
                val = np.clip(data[i, j], vmin, vmax)
                # Get color from colormap
                color_hex = colormap(val)
                # Convert hex to RGB
                r = int(color_hex[1:3], 16)
                g = int(color_hex[3:5], 16)
                b = int(color_hex[5:7], 16)
                rgba[i, j] = [r, g, b, 180]  # Semi-transparent

    return {
        "rgba": np.flipud(rgba),  # Flip for image coordinates
        "bounds": bounds,
        "vmin": vmin,
        "vmax": vmax,
    }


def visualise_nimrod(
    input_path: Path,
    shapefile_path: Path,
    aggregate: str = "mean",
    recursive: bool = False,
    output_html: Path | None = None,
    open_browser: bool = True,
) -> Path:
    """Build an interactive map showing NIMROD rainfall data.

    Parameters
    ----------
    input_path :
        Path to NetCDF file or directory containing NetCDF files.
    shapefile_path :
        Path to shapefile for boundary display.
    aggregate :
        Aggregation method if input is a directory: "mean", "max", "min", or "sum".
    recursive :
        If True and input is a directory, search recursively for NetCDF files.
    output_html :
        Where to write the HTML file. A temporary file is used when None.
    open_browser :
        If True, automatically open the HTML file in the default browser.

    Returns
    -------
    Path
        The path to the written HTML file.
    """
    logger.info(f"Loading NetCDF data from {input_path}")
    cube = load_and_aggregate_netcdf(input_path, aggregate=aggregate, recursive=recursive)

    logger.info(f"Loading shapefile from {shapefile_path}")
    gdf = load_shapefile(shapefile_path)

    # Get data statistics (excluding NaN)
    data = cube.data

    # Convert masked array to regular array with NaN for masked values
    if np.ma.is_masked(data):
        data = np.ma.filled(data, np.nan)

    valid_data = data[~np.isnan(data)]

    if valid_data.size == 0:
        raise ValueError("All data values are NaN, cannot create visualization")

    data_min = float(np.min(valid_data))
    data_max = float(np.max(valid_data))
    data_mean = float(np.mean(valid_data))

    # Handle edge case where all values are the same
    if data_min == data_max:
        data_max = data_min + 1e-10

    logger.info(f"Rainfall data: min={data_min:.2f}, max={data_max:.2f}, mean={data_mean:.2f}")

    # Create colormap (blue to red for rainfall)
    colormap = cm.LinearColormap(
        colors=[
            "#f7fbff",
            "#deebf7",
            "#c6dbef",
            "#9ecae1",
            "#6baed6",
            "#4292c6",
            "#2171b5",
            "#08519c",
            "#08306b",
        ],
        vmin=data_min,
        vmax=data_max,
        caption="Rainfall (mm/hr)",
    )

    # Get map center from shapefile
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # Create Folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
    )

    # Add shapefile boundary
    folium.GeoJson(
        gdf,
        name="Region Boundary",
        style_function=lambda x: {
            "fillColor": "none",
            "color": "#ff0000",
            "weight": 2,
            "fillOpacity": 0,
        },
    ).add_to(m)

    # Convert cube to overlay data
    overlay_data = cube_to_leaflet_overlay(cube, colormap, data_min, data_max)

    # Create a PNG image from the RGBA array
    import base64
    import io

    from PIL import Image

    img = Image.fromarray(overlay_data["rgba"], mode="RGBA")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    img_url = f"data:image/png;base64,{img_str}"

    # Add image overlay
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=overlay_data["bounds"],
        opacity=0.6,
        name="Rainfall",
    ).add_to(m)

    # Add colormap legend
    colormap.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Fit map to shapefile bounds
    sw = [bounds[1], bounds[0]]  # lat, lon
    ne = [bounds[3], bounds[2]]
    m.fit_bounds([sw, ne], padding=[30, 30])

    # Add title
    title_html = f"""
    <div style="position: fixed;
                top: 10px; left: 50px; width: 400px; height: 60px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <b>NIMROD Rainfall Data</b><br>
    Min: {data_min:.2f} mm/hr | Max: {data_max:.2f} mm/hr | Mean: {data_mean:.2f} mm/hr
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Write HTML
    if output_html is not None:
        html_path = Path(output_html).resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", prefix="nimrod_map_", delete=False)
        tmp.close()
        html_path = Path(tmp.name).resolve()

    m.save(str(html_path))
    logger.info(f"Map saved to {html_path}")

    if open_browser:
        webbrowser.open(html_path.as_uri())

    return html_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Visualise NIMROD rainfall data on an interactive map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to NetCDF file or directory containing NetCDF files.",
    )
    p.add_argument(
        "--shapefile",
        type=Path,
        required=True,
        help="Path to shapefile for boundary display.",
    )
    p.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "max", "min", "sum"],
        help="Aggregation method when input is a directory (default: mean).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search for NetCDF files recursively in subdirectories.",
    )
    p.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the HTML map in the default browser.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the HTML map. A temporary file is used when omitted.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    if not args.shapefile.exists():
        logger.error(f"Shapefile not found: {args.shapefile}")
        sys.exit(1)

    try:
        visualise_nimrod(
            input_path=args.input,
            shapefile_path=args.shapefile,
            aggregate=args.aggregate,
            recursive=args.recursive,
            output_html=args.output,
            open_browser=not args.no_open,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.error(f"Fatal error: {exc}", exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
