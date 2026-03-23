"""
topography.py

Land topography and slope data access for the eoflow package.

This module provides three complementary capabilities:

1. **Topography** – extract elevation data from a Digital Elevation Model
   (DEM) for a given polygon, returning a structured xarray DataArray on a
   British National Grid (EPSG:27700) regular grid.

2. **Slope magnitude** – compute the terrain slope for a given polygon using
   the Horn (1981) 3×3 neighbourhood method, which is identical to the
   algorithm used by ArcGIS Spatial Analyst.

3. **Slope direction (aspect)** – compute the compass bearing of the steepest
   uphill direction for each cell using the same Horn (1981) partial
   derivatives, expressed as degrees or radians clockwise from North.

All three functions accept a Shapely geometry in WGS-84 (EPSG:4326) and a
path to a GeoTIFF DEM.  They reproject the raster onto a BNG grid aligned
with the polygon's extent, mask all cells outside the polygon to ``NaN``, and
return an :class:`xarray.DataArray`.

Slope algorithm
---------------
Following Horn (1981) and the ArcGIS Spatial Analyst `Slope` tool
implementation (planar method).

For each central cell ``e`` in the 3 × 3 neighbourhood::

    a  b  c
    d  e  f
    g  h  i

.. math::

    \\frac{dz}{dx} = \\frac{(c + 2f + i) - (a + 2d + g)}{8 \\cdot \\Delta x}

    \\frac{dz}{dy} = \\frac{(g + 2h + i) - (a + 2b + c)}{8 \\cdot \\Delta y}

    \\text{slope}_{\\text{rad}} = \\arctan\\!\\sqrt{\\left(\\frac{dz}{dx}\\right)^2
                                  + \\left(\\frac{dz}{dy}\\right)^2}

where :math:`\\Delta x` and :math:`\\Delta y` are the cell sizes in the
easting and northing directions respectively (metres on BNG).

The slope magnitude is reported in **degrees** by default but can also be
expressed as **percent rise** or **radians**.

Aspect algorithm
----------------
The aspect (slope direction) is derived from the same Horn (1981) partial
derivatives.  The gradient vector :math:`(dz/dx,\\, dz/dy)` points in the
direction of steepest ascent in BNG Cartesian space (East = +x, North = +y).
The aspect is the compass bearing of that vector, measured **clockwise from
North**:

.. math::

    \\text{aspect} = \\left(90^{\\circ}
                     - \\arctan2\\!\\left(\\frac{dz}{dy},\\,\\frac{dz}{dx}\\right)
                     \\right) \\bmod 360^{\\circ}

Cells where both partial derivatives are zero (flat terrain) are set to
``NaN``.

References
----------
Horn, B. K. P. (1981). Hill shading and the reflectance map.
*Proceedings of the IEEE*, 69(1), 14–47.
https://doi.org/10.1109/PROC.1981.11918

ArcGIS Pro Tool Reference – Slope (Spatial Analyst):
https://pro.arcgis.com/en/pro-app/3.4/tool-reference/spatial-analyst/slope.htm

ArcGIS Pro Tool Reference – Aspect (Spatial Analyst):
https://pro.arcgis.com/en/pro-app/3.4/tool-reference/spatial-analyst/aspect.htm
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import xarray

import numpy as np
import rasterio
import rasterio.transform
import rasterio.warp
from rasterio.crs import CRS

from eoflow.log_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default output raster resolution in metres (BNG).
DEFAULT_RESOLUTION_M = 50

#: British National Grid CRS (EPSG:27700).
BNG_CRS = CRS.from_epsg(27700)

#: WGS-84 CRS (EPSG:4326).
WGS84_CRS = CRS.from_epsg(4326)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _polygon_to_bng(polygon):
    """Reproject a WGS-84 Shapely Polygon/MultiPolygon to BNG (EPSG:27700).

    Parameters
    ----------
    polygon :
        Input geometry in WGS-84 (EPSG:4326).

    Returns
    -------
    shapely.geometry.Polygon or MultiPolygon
        The reprojected geometry in EPSG:27700 (coordinates in BNG metres).
    """
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    return shapely_transform(transformer.transform, polygon)


def _load_dem_for_polygon(
    dem_path: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    res: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a DEM, warp it onto a BNG grid, and resample to the target resolution.

    The source DEM may be in any CRS; it is reprojected to EPSG:27700 using
    bilinear resampling before being returned.

    Parameters
    ----------
    dem_path :
        Path to the source DEM GeoTIFF (any CRS).
    x_min, x_max :
        Easting extent in BNG metres (EPSG:27700).
    y_min, y_max :
        Northing extent in BNG metres (EPSG:27700).
    res :
        Output cell size in metres.

    Returns
    -------
    elevation : numpy.ndarray, shape (H, W), dtype float32
        Elevation values in metres.  Cells outside the DEM coverage are NaN.
    eastings : numpy.ndarray, shape (W,)
        BNG easting coordinates of cell centres (west to east).
    northings : numpy.ndarray, shape (H,)
        BNG northing coordinates of cell centres (south to north).
    """
    # Build the target coordinate arrays.
    eastings = np.arange(x_min, x_max, res, dtype=np.float64)
    northings = np.arange(y_min, y_max, res, dtype=np.float64)

    width = len(eastings)
    height = len(northings)

    # rasterio.transform.from_origin expects the *top-left* corner of the
    # top-left cell and the cell size.  Our northings array runs south→north,
    # so y_max is the northern edge of the topmost row.
    dst_transform = rasterio.transform.from_origin(
        west=x_min,
        north=y_max,
        xsize=res,
        ysize=res,
    )

    # Output buffer filled with NaN.
    dst_array = np.full((height, width), np.nan, dtype=np.float32)

    with rasterio.open(str(dem_path)) as src:
        src_nodata = src.nodata
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=BNG_CRS,
            resampling=rasterio.warp.Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )

    # Replace any remaining explicit no-data fill values with NaN.
    if src_nodata is not None and not np.isnan(float(src_nodata)):
        dst_array[dst_array == np.float32(src_nodata)] = np.nan

    # rasterio writes rows top→bottom; flip so that row 0 is the southernmost
    # row, matching the south→north order of `northings`.
    dst_array = dst_array[::-1, :]

    return dst_array, eastings, northings


def _compute_aspect(
    elevation: np.ndarray,
    cell_size_x: float,
    cell_size_y: float,
    output: Literal["degrees", "radians"] = "degrees",
) -> np.ndarray:
    """Compute slope aspect (direction) using Horn's (1981) 3 × 3 neighbourhood.

    The aspect is the compass bearing of the steepest uphill direction,
    measured **clockwise from North** (0 = North, 90 = East, 180 = South,
    270 = West).  Flat cells (zero gradient magnitude) are set to ``NaN``.

    Parameters
    ----------
    elevation :
        2-D array of elevation values (metres).  NaN for no-data.
    cell_size_x :
        Cell width in the x (easting) direction (metres).
    cell_size_y :
        Cell height in the y (northing) direction (metres).
    output :
        Unit for the output aspect values:

        - ``"degrees"`` – compass bearing in degrees, range [0, 360).
        - ``"radians"`` – compass bearing in radians, range [0, 2π).

    Returns
    -------
    numpy.ndarray, shape same as *elevation*, dtype float32
        Aspect values.  Border cells, cells adjacent to missing elevation
        data, and flat cells are set to ``NaN``.

    Notes
    -----
    The 3 × 3 neighbourhood is labelled with row indices increasing
    **northward** (after the south-to-north array flip applied in
    :func:`_load_dem_for_polygon`)::

        a  b  c   <- southernmost row of the window
        d  e  f
        g  h  i   <- northernmost row of the window

    The Horn (1981) partial derivatives are therefore:

    .. math::

        \\frac{dz}{dx} = \\frac{(c + 2f + i) - (a + 2d + g)}{8 \\cdot \\Delta x}

        \\frac{dz}{dy} = \\frac{(g + 2h + i) - (a + 2b + c)}{8 \\cdot \\Delta y}

    where positive :math:`dz/dx` means terrain rises going East and positive
    :math:`dz/dy` means terrain rises going North.  The aspect is then:

    .. math::

        \\text{aspect} = \\left(90^{\\circ}
                         - \\arctan2\\!\\left(\\frac{dz}{dy},\\,\\frac{dz}{dx}\\right)
                         \\right) \\bmod 360^{\\circ}
    """
    nrows, ncols = elevation.shape
    aspect = np.full((nrows, ncols), np.nan, dtype=np.float32)

    if nrows < 3 or ncols < 3:
        # Grid too small to compute any aspect values.
        return aspect

    # Extract the eight neighbours via vectorised slicing.
    # Row indices increase northward, so row 0 is south and row -1 is north.
    a = elevation[0:-2, 0:-2]
    b = elevation[0:-2, 1:-1]
    c = elevation[0:-2, 2:]
    d = elevation[1:-1, 0:-2]
    # e = elevation[1:-1, 1:-1]  # central cell – not needed explicitly
    f = elevation[1:-1, 2:]
    g = elevation[2:, 0:-2]
    h = elevation[2:, 1:-1]
    i = elevation[2:, 2:]

    # Partial derivatives (Horn 1981) – identical to those in _compute_slope.
    # dz_dx > 0 ↔ terrain rises going East.
    # dz_dy > 0 ↔ terrain rises going North.
    dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * cell_size_x)
    dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * cell_size_y)

    # atan2(dz_dy, dz_dx) gives the uphill direction as a mathematical angle
    # (counter-clockwise from East, in radians).
    # Converting to compass bearing (clockwise from North):
    #   compass = (π/2 − math_angle) mod 2π
    aspect_rad = (np.pi / 2.0 - np.arctan2(dz_dy, dz_dx)) % (2.0 * np.pi)

    if output == "radians":
        result = aspect_rad.astype(np.float32)
    else:  # "degrees" (default)
        result = np.degrees(aspect_rad).astype(np.float32)

    # Flat cells (zero gradient) → NaN.
    flat_mask = (dz_dx == 0.0) & (dz_dy == 0.0)
    result[flat_mask] = np.nan

    aspect[1:-1, 1:-1] = result

    # Propagate NaN wherever any of the eight neighbours was NaN.
    nan_mask = (
        np.isnan(a)
        | np.isnan(b)
        | np.isnan(c)
        | np.isnan(d)
        | np.isnan(f)
        | np.isnan(g)
        | np.isnan(h)
        | np.isnan(i)
    )
    aspect[1:-1, 1:-1][nan_mask] = np.nan

    return aspect


def _compute_slope(
    elevation: np.ndarray,
    cell_size_x: float,
    cell_size_y: float,
    output: Literal["degrees", "percent_rise", "radians"] = "degrees",
) -> np.ndarray:
    """Compute slope using Horn's (1981) 3 × 3 neighbourhood method.

    This matches the ArcGIS Spatial Analyst *Slope* tool (planar method).

    Parameters
    ----------
    elevation :
        2-D array of elevation values (metres).  NaN for no-data.
    cell_size_x :
        Cell width in the x (easting) direction (metres).
    cell_size_y :
        Cell height in the y (northing) direction (metres).
    output :
        Unit for the output slope values:

        - ``"degrees"``      – angle from horizontal in degrees (0–90).
        - ``"percent_rise"`` – rise / run × 100 (0–∞).
        - ``"radians"``      – angle from horizontal in radians.

    Returns
    -------
    numpy.ndarray, shape same as *elevation*, dtype float32
        Slope values.  Border cells and cells adjacent to missing elevation
        data are set to NaN.

    Notes
    -----
    The 3 × 3 neighbourhood is labelled::

        a  b  c
        d  e  f
        g  h  i

    Following Horn (1981) and the ArcGIS Spatial Analyst documentation:

    .. math::

        \\frac{dz}{dx} = \\frac{(c + 2f + i) - (a + 2d + g)}{8 \\cdot \\Delta x}

        \\frac{dz}{dy} = \\frac{(g + 2h + i) - (a + 2b + c)}{8 \\cdot \\Delta y}

        \\text{slope}_{\\text{rad}} = \\arctan\\!\\sqrt{\\left(\\frac{dz}{dx}\\right)^2
                                      + \\left(\\frac{dz}{dy}\\right)^2}

    The outermost ring of output cells is always NaN because there are not
    enough neighbours to form a complete 3 × 3 window.
    """
    nrows, ncols = elevation.shape
    slope = np.full((nrows, ncols), np.nan, dtype=np.float32)

    if nrows < 3 or ncols < 3:
        # Grid too small to compute any slope values.
        return slope

    # Extract the eight neighbours via vectorised slicing.
    # Valid central cells are at rows 1:-1, cols 1:-1.
    a = elevation[0:-2, 0:-2]
    b = elevation[0:-2, 1:-1]
    c = elevation[0:-2, 2:]
    d = elevation[1:-1, 0:-2]
    # e = elevation[1:-1, 1:-1]  # central cell – not needed explicitly
    f = elevation[1:-1, 2:]
    g = elevation[2:, 0:-2]
    h = elevation[2:, 1:-1]
    i = elevation[2:, 2:]

    # Partial derivatives (Horn 1981).
    dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * cell_size_x)
    dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * cell_size_y)

    # Rise / run magnitude.
    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)

    if output == "radians":
        slope[1:-1, 1:-1] = np.arctan(rise_run).astype(np.float32)
    elif output == "percent_rise":
        # tan(arctan(rise_run)) == rise_run, so this simplifies to × 100.
        slope[1:-1, 1:-1] = (rise_run * 100.0).astype(np.float32)
    else:  # "degrees" (default)
        slope[1:-1, 1:-1] = np.degrees(np.arctan(rise_run)).astype(np.float32)

    # Propagate NaN wherever any of the eight neighbours was NaN.
    nan_mask = (
        np.isnan(a)
        | np.isnan(b)
        | np.isnan(c)
        | np.isnan(d)
        | np.isnan(f)
        | np.isnan(g)
        | np.isnan(h)
        | np.isnan(i)
    )
    slope[1:-1, 1:-1][nan_mask] = np.nan

    return slope


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_topography_for_polygon(
    polygon,
    dem_path: Path | str,
    *,
    res: int = DEFAULT_RESOLUTION_M,
) -> "xarray.DataArray":
    """Extract elevation data clipped to a polygon from a DEM.

    Reads the DEM at *dem_path*, reprojects it onto a British National Grid
    (EPSG:27700) regular grid bounded by *polygon*'s extent, and masks all
    grid cells whose centre lies **outside** *polygon* to ``NaN``.

    Parameters
    ----------
    polygon :
        Area of interest in **WGS-84 (EPSG:4326)**.  May be a
        ``shapely.geometry.Polygon`` or ``MultiPolygon`` – for example, a
        catchment boundary returned by
        :func:`eoflow.catchment.delineate_catchment`.
    dem_path :
        Path to a GeoTIFF DEM (any CRS; will be reprojected to BNG via
        bilinear resampling).
    res :
        Output grid resolution in BNG metres (default:
        :data:`DEFAULT_RESOLUTION_M` = 50 m).

    Returns
    -------
    xarray.DataArray
        Elevation in **metres** with dimensions ``(y, x)``:

        - ``y`` – BNG northings (metres, EPSG:27700), south-to-north.
        - ``x`` – BNG eastings (metres, EPSG:27700), west-to-east.

        Cells whose centre lies **outside** *polygon* are ``NaN``.

        ``attrs`` on the returned array:

        - ``units``      ``"m"``
        - ``long_name``  ``"Elevation"``
        - ``crs``        ``"EPSG:27700"``
        - ``source``     absolute path to the DEM file

    Raises
    ------
    FileNotFoundError
        If *dem_path* does not exist.
    ValueError
        If the reprojected polygon has zero area.

    Examples
    --------
    ::

        from shapely.geometry import box
        from eoflow.topography import get_topography_for_polygon

        # Small bounding box over Dartmoor
        dartmoor = box(-3.9, 50.5, -3.7, 50.7)
        da = get_topography_for_polygon(
            dartmoor,
            dem_path="data/devon_dem.tif",
        )
        print(da)
        # <xarray.DataArray 'elevation' (y: 44, x: 22)>
        # Coordinates:
        #   * y    (y) float64 ...
        #   * x    (x) float64 ...
    """
    import shapely
    import xarray as xr

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    # --- Reproject polygon to BNG and derive snapped extraction extent -----
    polygon_bng = _polygon_to_bng(polygon)
    if polygon_bng.area == 0.0:
        raise ValueError("Input polygon has zero area after reprojection to BNG.")

    x_min_bb, y_min_bb, x_max_bb, y_max_bb = polygon_bng.bounds

    # Snap to the resolution grid for consistent pixel alignment.
    x_min = float(np.floor(x_min_bb / res) * res)
    y_min = float(np.floor(y_min_bb / res) * res)
    x_max = float((np.ceil(x_max_bb / res) + 1) * res)
    y_max = float((np.ceil(y_max_bb / res) + 1) * res)

    logger.info("get_topography_for_polygon")
    logger.info(
        "  Polygon BNG bounds : %.0f, %.0f → %.0f, %.0f",
        x_min_bb,
        y_min_bb,
        x_max_bb,
        y_max_bb,
    )
    logger.info(
        "  Extraction extent  : x=[%.0f, %.0f]  y=[%.0f, %.0f]  res=%d m",
        x_min,
        x_max,
        y_min,
        y_max,
        res,
    )

    # --- Load and warp DEM -------------------------------------------------
    elevation, eastings, northings = _load_dem_for_polygon(
        dem_path, x_min, x_max, y_min, y_max, res
    )

    # --- Build polygon mask and apply --------------------------------------
    xx, yy = np.meshgrid(eastings, northings)
    grid_points = shapely.points(xx.ravel(), yy.ravel())
    polygon_mask = shapely.within(grid_points, polygon_bng).reshape(xx.shape)

    n_inside = int(polygon_mask.sum())
    n_total = polygon_mask.size
    logger.info(
        "  Polygon covers %d / %d pixel(s) (%.1f%%) on the BNG grid.",
        n_inside,
        n_total,
        100.0 * n_inside / n_total if n_total > 0 else 0.0,
    )

    elevation[~polygon_mask] = np.nan

    # --- Assemble xarray DataArray ----------------------------------------
    da = xr.DataArray(
        data=elevation,
        coords={"y": northings, "x": eastings},
        dims=["y", "x"],
        name="elevation",
        attrs={
            "units": "m",
            "long_name": "Elevation",
            "crs": "EPSG:27700",
            "source": str(dem_path.resolve()),
        },
    )

    logger.info(
        "Assembled elevation DataArray: grid %d (y) × %d (x).",
        len(northings),
        len(eastings),
    )

    return da


def get_slope_for_polygon(
    polygon,
    dem_path: Path | str,
    *,
    res: int = DEFAULT_RESOLUTION_M,
    output: Literal["degrees", "percent_rise", "radians"] = "degrees",
) -> "xarray.DataArray":
    """Compute terrain slope clipped to a polygon from a DEM.

    Uses Horn's (1981) 3 × 3 neighbourhood method – the same algorithm as
    the ArcGIS Spatial Analyst *Slope* tool (planar method).

    The DEM is reprojected to British National Grid (EPSG:27700) before the
    slope is computed so that the x and y cell sizes are both expressed in
    metres, giving correct results regardless of the DEM's source CRS.

    Parameters
    ----------
    polygon :
        Area of interest in **WGS-84 (EPSG:4326)**.  May be a
        ``shapely.geometry.Polygon`` or ``MultiPolygon`` – for example, a
        catchment boundary returned by
        :func:`eoflow.catchment.delineate_catchment`.
    dem_path :
        Path to a GeoTIFF DEM (any CRS; will be reprojected to BNG via
        bilinear resampling).
    res :
        Output grid resolution in BNG metres (default:
        :data:`DEFAULT_RESOLUTION_M` = 50 m).
    output :
        Units for the slope output:

        - ``"degrees"``      – angle from horizontal in degrees (0–90).
        - ``"percent_rise"`` – rise / run × 100 (0–∞).
        - ``"radians"``      – angle from horizontal in radians.

    Returns
    -------
    xarray.DataArray
        Slope with dimensions ``(y, x)``:

        - ``y`` – BNG northings (metres, EPSG:27700), south-to-north.
        - ``x`` – BNG eastings (metres, EPSG:27700), west-to-east.

        Cells whose centre lies **outside** *polygon*, outermost border cells
        (which lack a full 3 × 3 neighbourhood), and cells adjacent to
        missing elevation data are ``NaN``.

        ``attrs`` on the returned array:

        - ``units``      one of ``"degrees"``, ``"percent"``, or
          ``"radians"``
        - ``long_name``  ``"Slope"``
        - ``crs``        ``"EPSG:27700"``
        - ``source``     absolute path to the DEM file
        - ``method``     ``"Horn (1981) 3x3 neighbourhood"``

    Raises
    ------
    FileNotFoundError
        If *dem_path* does not exist.
    ValueError
        If the reprojected polygon has zero area, or if *output* is not
        one of the recognised strings.

    Examples
    --------
    ::

        from shapely.geometry import box
        from eoflow.topography import get_slope_for_polygon

        dartmoor = box(-3.9, 50.5, -3.7, 50.7)
        da = get_slope_for_polygon(
            dartmoor,
            dem_path="data/devon_dem.tif",
            output="degrees",
        )
        print(da.max().item(), "°")   # maximum slope angle in the polygon

        # Mean slope as percent rise
        da_pct = get_slope_for_polygon(
            dartmoor,
            dem_path="data/devon_dem.tif",
            output="percent_rise",
        )
        print(da_pct.mean().item(), "%")
    """
    import shapely
    import xarray as xr

    valid_outputs = {"degrees", "percent_rise", "radians"}
    if output not in valid_outputs:
        raise ValueError(f"'output' must be one of {sorted(valid_outputs)!r}, got {output!r}.")

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    # --- Reproject polygon to BNG and derive snapped extraction extent -----
    polygon_bng = _polygon_to_bng(polygon)
    if polygon_bng.area == 0.0:
        raise ValueError("Input polygon has zero area after reprojection to BNG.")

    x_min_bb, y_min_bb, x_max_bb, y_max_bb = polygon_bng.bounds

    # Add a one-cell buffer around the snapped extent so that the slope
    # kernel has valid neighbours for every cell that falls inside the polygon,
    # including those on the outer edge of the extraction window.
    x_min_buf = float(np.floor(x_min_bb / res) * res) - res
    y_min_buf = float(np.floor(y_min_bb / res) * res) - res
    x_max_buf = float((np.ceil(x_max_bb / res) + 1) * res) + res
    y_max_buf = float((np.ceil(y_max_bb / res) + 1) * res) + res

    logger.info("get_slope_for_polygon")
    logger.info(
        "  Polygon BNG bounds : %.0f, %.0f → %.0f, %.0f",
        x_min_bb,
        y_min_bb,
        x_max_bb,
        y_max_bb,
    )
    logger.info(
        "  Buffered extent    : x=[%.0f, %.0f]  y=[%.0f, %.0f]  res=%d m",
        x_min_buf,
        x_max_buf,
        y_min_buf,
        y_max_buf,
        res,
    )
    logger.info("  Slope output units : %s", output)

    # --- Load and warp DEM (with buffer) -----------------------------------
    elevation, eastings_buf, northings_buf = _load_dem_for_polygon(
        dem_path, x_min_buf, x_max_buf, y_min_buf, y_max_buf, res
    )

    # --- Compute slope on the full buffered grid ---------------------------
    slope_buf = _compute_slope(
        elevation,
        cell_size_x=float(res),
        cell_size_y=float(res),
        output=output,
    )

    # --- Build polygon mask on the buffered grid and apply -----------------
    xx, yy = np.meshgrid(eastings_buf, northings_buf)
    grid_points = shapely.points(xx.ravel(), yy.ravel())
    polygon_mask = shapely.within(grid_points, polygon_bng).reshape(xx.shape)

    slope_buf[~polygon_mask] = np.nan

    # --- Crop to the original (un-buffered) snapped extent -----------------
    x_min_out = float(np.floor(x_min_bb / res) * res)
    y_min_out = float(np.floor(y_min_bb / res) * res)
    x_max_out = float((np.ceil(x_max_bb / res) + 1) * res)
    y_max_out = float((np.ceil(y_max_bb / res) + 1) * res)

    x_mask = (eastings_buf >= x_min_out) & (eastings_buf < x_max_out)
    y_mask = (northings_buf >= y_min_out) & (northings_buf < y_max_out)

    slope_out = slope_buf[np.ix_(y_mask, x_mask)]
    eastings_out = eastings_buf[x_mask]
    northings_out = northings_buf[y_mask]

    # Map output identifier to CF-convention units string.
    unit_map = {
        "degrees": "degrees",
        "percent_rise": "percent",
        "radians": "radians",
    }

    # --- Assemble xarray DataArray ----------------------------------------
    da = xr.DataArray(
        data=slope_out,
        coords={"y": northings_out, "x": eastings_out},
        dims=["y", "x"],
        name="slope",
        attrs={
            "units": unit_map[output],
            "long_name": "Slope",
            "crs": "EPSG:27700",
            "source": str(dem_path.resolve()),
            "method": "Horn (1981) 3x3 neighbourhood",
        },
    )

    logger.info(
        "Assembled slope DataArray: grid %d (y) × %d (x).",
        len(northings_out),
        len(eastings_out),
    )

    return da


def get_aspect_for_polygon(
    polygon,
    dem_path: Path | str,
    *,
    res: int = DEFAULT_RESOLUTION_M,
    output: Literal["degrees", "radians"] = "degrees",
) -> "xarray.DataArray":
    """Compute terrain aspect (slope direction) clipped to a polygon from a DEM.

    Uses Horn's (1981) 3 × 3 neighbourhood method to derive the compass
    bearing of the steepest uphill direction for each grid cell.  The aspect
    is measured **clockwise from North** (0 = North, 90 = East, 180 = South,
    270 = West).

    The DEM is reprojected to British National Grid (EPSG:27700) before the
    aspect is computed so that the x and y cell sizes are both expressed in
    metres, giving correct results regardless of the DEM's source CRS.

    Parameters
    ----------
    polygon :
        Area of interest in **WGS-84 (EPSG:4326)**.  May be a
        ``shapely.geometry.Polygon`` or ``MultiPolygon`` – for example, a
        catchment boundary returned by
        :func:`eoflow.catchment.delineate_catchment`.
    dem_path :
        Path to a GeoTIFF DEM (any CRS; will be reprojected to BNG via
        bilinear resampling).
    res :
        Output grid resolution in BNG metres (default:
        :data:`DEFAULT_RESOLUTION_M` = 50 m).
    output :
        Units for the aspect output:

        - ``"degrees"`` – compass bearing in degrees, range [0, 360).
        - ``"radians"`` – compass bearing in radians, range [0, 2π).

    Returns
    -------
    xarray.DataArray
        Aspect with dimensions ``(y, x)``:

        - ``y`` – BNG northings (metres, EPSG:27700), south-to-north.
        - ``x`` – BNG eastings (metres, EPSG:27700), west-to-east.

        Cells whose centre lies **outside** *polygon*, outermost border cells
        (which lack a full 3 × 3 neighbourhood), cells adjacent to missing
        elevation data, and flat cells (zero gradient) are ``NaN``.

        ``attrs`` on the returned array:

        - ``units``      ``"degrees"`` or ``"radians"``
        - ``long_name``  ``"Aspect"``
        - ``crs``        ``"EPSG:27700"``
        - ``source``     absolute path to the DEM file
        - ``method``     ``"Horn (1981) 3x3 neighbourhood"``
        - ``convention`` ``"clockwise from North"``

    Raises
    ------
    FileNotFoundError
        If *dem_path* does not exist.
    ValueError
        If the reprojected polygon has zero area, or if *output* is not
        one of the recognised strings.

    Examples
    --------
    ::

        from shapely.geometry import box
        from eoflow.topography import get_aspect_for_polygon

        dartmoor = box(-3.9, 50.5, -3.7, 50.7)
        da = get_aspect_for_polygon(
            dartmoor,
            dem_path="data/devon_dem.tif",
            output="degrees",
        )
        # Mean aspect – roughly which compass direction the area faces
        print(da.mean().item(), "°")

        # Radians output
        da_rad = get_aspect_for_polygon(
            dartmoor,
            dem_path="data/devon_dem.tif",
            output="radians",
        )
    """
    import shapely
    import xarray as xr

    valid_outputs = {"degrees", "radians"}
    if output not in valid_outputs:
        raise ValueError(f"'output' must be one of {sorted(valid_outputs)!r}, got {output!r}.")

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    # --- Reproject polygon to BNG and derive snapped extraction extent -----
    polygon_bng = _polygon_to_bng(polygon)
    if polygon_bng.area == 0.0:
        raise ValueError("Input polygon has zero area after reprojection to BNG.")

    x_min_bb, y_min_bb, x_max_bb, y_max_bb = polygon_bng.bounds

    # Add a one-cell buffer so that every cell inside the polygon has a full
    # 3 × 3 neighbourhood available for the aspect kernel.
    x_min_buf = float(np.floor(x_min_bb / res) * res) - res
    y_min_buf = float(np.floor(y_min_bb / res) * res) - res
    x_max_buf = float((np.ceil(x_max_bb / res) + 1) * res) + res
    y_max_buf = float((np.ceil(y_max_bb / res) + 1) * res) + res

    logger.info("get_aspect_for_polygon")
    logger.info(
        "  Polygon BNG bounds : %.0f, %.0f → %.0f, %.0f",
        x_min_bb,
        y_min_bb,
        x_max_bb,
        y_max_bb,
    )
    logger.info(
        "  Buffered extent    : x=[%.0f, %.0f]  y=[%.0f, %.0f]  res=%d m",
        x_min_buf,
        x_max_buf,
        y_min_buf,
        y_max_buf,
        res,
    )
    logger.info("  Aspect output units : %s", output)

    # --- Load and warp DEM (with buffer) -----------------------------------
    elevation, eastings_buf, northings_buf = _load_dem_for_polygon(
        dem_path, x_min_buf, x_max_buf, y_min_buf, y_max_buf, res
    )

    # --- Compute aspect on the full buffered grid --------------------------
    aspect_buf = _compute_aspect(
        elevation,
        cell_size_x=float(res),
        cell_size_y=float(res),
        output=output,
    )

    # --- Build polygon mask on the buffered grid and apply -----------------
    xx, yy = np.meshgrid(eastings_buf, northings_buf)
    grid_points = shapely.points(xx.ravel(), yy.ravel())
    polygon_mask = shapely.within(grid_points, polygon_bng).reshape(xx.shape)

    aspect_buf[~polygon_mask] = np.nan

    # --- Crop to the original (un-buffered) snapped extent -----------------
    x_min_out = float(np.floor(x_min_bb / res) * res)
    y_min_out = float(np.floor(y_min_bb / res) * res)
    x_max_out = float((np.ceil(x_max_bb / res) + 1) * res)
    y_max_out = float((np.ceil(y_max_bb / res) + 1) * res)

    x_mask = (eastings_buf >= x_min_out) & (eastings_buf < x_max_out)
    y_mask = (northings_buf >= y_min_out) & (northings_buf < y_max_out)

    aspect_out = aspect_buf[np.ix_(y_mask, x_mask)]
    eastings_out = eastings_buf[x_mask]
    northings_out = northings_buf[y_mask]

    # --- Assemble xarray DataArray ----------------------------------------
    da = xr.DataArray(
        data=aspect_out,
        coords={"y": northings_out, "x": eastings_out},
        dims=["y", "x"],
        name="aspect",
        attrs={
            "units": output,
            "long_name": "Aspect",
            "crs": "EPSG:27700",
            "source": str(dem_path.resolve()),
            "method": "Horn (1981) 3x3 neighbourhood",
            "convention": "clockwise from North",
        },
    )

    logger.info(
        "Assembled aspect DataArray: grid %d (y) × %d (x).",
        len(northings_out),
        len(eastings_out),
    )

    return da
