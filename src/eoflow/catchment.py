from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import rasterio
from shapely.geometry import Point, Polygon, shape

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
from pysheds.sview import View  # noqa: E402

logger = get_logger(__name__)


def _delineate_catchment_core(
    point: Point,
    dem_path: Union[str, Path],
    dirmap: Tuple[int, int, int, int, int, int, int, int] = (64, 128, 1, 2, 4, 8, 16, 32),
    apply_input_mask: bool = False,
    nodata_in: Optional[float] = None,
    nodata_out: Optional[float] = None,
    pit_fill: bool = True,
    pit_fill_epsilon: float = 0.0001,
    fill_depressions: bool = True,
    resolve_flats: bool = True,
    routing: str = "d8",
    flow_acc_threshold: int = 5000,
) -> Tuple[Polygon, Point, float]:
    """
    Core catchment delineation returning polygon, snapped pour point, and flow accumulation.

    This is the internal workhorse used by :func:`delineate_catchment` and
    :func:`delineate_catchment_with_metadata`.  It returns a tuple of
    ``(catchment_polygon, snapped_pour_point, flow_acc_at_pour_point)`` so
    callers can see where the pour point was snapped to on the stream network
    and how much upstream area drains to it.

    See :func:`delineate_catchment` for full parameter documentation.

    Returns:
        A tuple of (Polygon, Point, float) where:
            - Polygon is the delineated catchment boundary
            - Point is the snapped pour point on the stream network
            - float is the flow-accumulation value (upstream cell count) at
              the snapped pour point; ``nan`` if it could not be determined
    """
    # Validate inputs
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    # Initialize Grid
    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))

    # Get the coordinate system info
    try:
        x, y = point.x, point.y
    except AttributeError:
        raise ValueError("Invalid point geometry. Must be a Shapely Point object.")

    # Check if point is within DEM bounds
    if not (grid.bbox[0] <= x <= grid.bbox[2] and grid.bbox[1] <= y <= grid.bbox[3]):
        raise ValueError(
            f"Point ({x}, {y}) is outside DEM bounds: "
            f"x: [{grid.bbox[0]}, {grid.bbox[2]}], y: [{grid.bbox[1]}, {grid.bbox[3]}]"
        )

    # Build optional nodata kwargs – only include them when the caller
    # explicitly provided a value so that pysheds can fall back to its own
    # per-step defaults and avoid dtype conflicts (e.g. float nodata on a
    # uint8 flow-direction array).
    _nodata_in_kw = {} if nodata_in is None else {"nodata_in": nodata_in}
    _nodata_out_kw = {} if nodata_out is None else {"nodata_out": nodata_out}

    # Detect when the DEM's nodata is a float that cannot be safely cast
    # into integer arrays.  pysheds propagates the ViewFinder nodata to
    # every output Raster and checks ``np.can_cast(nodata, dtype, 'safe')``.
    # Under NumPy 2.x (NEP 50) float → int is *never* a safe cast, so
    # steps that produce integer arrays (flowdir, accumulation, catchment)
    # need an explicit integer nodata_out.
    # After fill_pits the DEM is upcast to float64; pysheds propagates that
    # float nodata into every downstream Raster.  Under NumPy 2.x (NEP 50),
    # np.can_cast raises TypeError for Python scalars, and a float nodata
    # cannot be safely cast into integer (flowdir/accumulation) or boolean
    # (catchment) arrays.  We always supply explicit NumPy scalar nodata_out
    # values to avoid the TypeError regardless of the source DEM dtype.
    _int_nodata_out = {"nodata_out": np.int64(0)}  # for flowdir & accumulation
    _bool_nodata_out = {"nodata_out": np.bool_(False)}  # for catchment
    logger.debug("Using nodata_out=int64(0) for integer steps, bool_(False) for catchment")

    # Step 1: Fill pits in DEM
    if pit_fill:
        logger.debug("Filling pits in DEM")
        pit_filled_dem = grid.fill_pits(dem, **_nodata_in_kw, **_nodata_out_kw)

        # Step 2: Fill depressions (multi-cell sinks larger than single pits)
        if fill_depressions:
            flooded_dem = grid.fill_depressions(pit_filled_dem)
        else:
            flooded_dem = pit_filled_dem

        # Step 3: Resolve flats
        if resolve_flats:
            inflated_dem = grid.resolve_flats(flooded_dem, eps=pit_fill_epsilon, **_nodata_out_kw)
        else:
            inflated_dem = flooded_dem
    else:
        inflated_dem = dem

    # Step 3: Compute flow direction
    if routing.lower() == "d8":
        logger.debug("Computing flow direction")
        fdir = grid.flowdir(
            inflated_dem,
            dirmap=dirmap,
            routing=routing,
            **_int_nodata_out,
        )
    else:
        raise ValueError(f"Unsupported routing method: {routing}. Use 'd8' or 'dinf'.")

    # Step 4: Compute flow accumulation
    logger.debug("Computing flow accumulation")
    acc = grid.accumulation(
        fdir,
        dirmap=dirmap,
        routing=routing,
        apply_input_mask=apply_input_mask,
        **_int_nodata_out,
    )

    # Step 5: Snap pour point to nearest high-accumulation cell
    # This ensures the pour point is on a stream rather than a hillslope
    #
    # NOTE: We deliberately avoid ``grid.snap_to_mask`` here. Under NumPy 2.x
    # (NEP 50), that method fails unconditionally: it internally calls
    # ``self.view(mask, ..., nodata=False, dtype=np.bool_)`` using the *Python*
    # bool literal ``False`` rather than ``np.bool_(False)``, and
    # ``np.can_cast`` raises ``TypeError`` for Python scalars. This makes
    # every snap attempt fail and silently fall back to the raw, unsnapped
    # pour point (producing near-zero catchments). We instead call the
    # lower-level ``View.snap_to_mask`` directly on a plain boolean array,
    # which performs the same nearest-neighbour (KD-tree) lookup without
    # going through the broken ``view()``/nodata-casting code path.
    try:
        logger.debug("Snapping pour point to nearest high-accumulation cell")
        mask = np.asarray(acc) > flow_acc_threshold
        x_snap, y_snap = View.snap_to_mask(mask, (x, y), affine=grid.affine, return_dist=False)
        x_snap, y_snap = float(x_snap), float(y_snap)
        snap_dist = ((x_snap - x) ** 2 + (y_snap - y) ** 2) ** 0.5
        logger.info(
            "  Pour point snapped: (%.6f, %.6f) → (%.6f, %.6f)  offset=%.6f°",
            x,
            y,
            x_snap,
            y_snap,
            snap_dist,
        )
    except Exception as e:
        # If snapping fails, try using the original point
        logger.warning("Could not snap point to stream network: %s", e)
        logger.warning("Using original point coordinates: (%.6f, %.6f)", x, y)
        x_snap, y_snap = x, y

    # Retrieve the flow-accumulation value at the (snapped) pour point
    try:
        logger.debug("Retrieving flow-accumulation value at pour point")
        col, row = grid.nearest_cell(x_snap, y_snap)
        # nearest_cell may return numpy scalars or 0-d arrays; use int() to be safe
        acc_val = np.array(acc)[int(row), int(col)]
        flow_acc_at_pour_point = float(np.asarray(acc_val).flat[0])
    except Exception:
        flow_acc_at_pour_point = float("nan")
        logger.debug("Could not read flow-accumulation at pour point; storing nan")

    # Step 6: Delineate the catchment
    # catchment() produces a boolean raster; use np.bool_(False) as nodata_out
    # so that pysheds' NEP-50-aware can_cast check succeeds.
    try:
        logger.debug("Delineating catchment from pour point (%.6f, %.6f)", x_snap, y_snap)
        catch = grid.catchment(
            x=x_snap,
            y=y_snap,
            fdir=fdir,
            dirmap=dirmap,
            routing=routing,
            xytype="coordinate",
            **_bool_nodata_out,
        )
    except Exception as e:
        raise ValueError(f"Failed to delineate catchment: {e}")

    # Step 7: Convert catchment raster to polygon
    # Extract the catchment boundary as a polygon
    try:
        logger.debug("Converting catchment raster to polygon")
        # Get the shapes (polygons) from the catchment raster
        shapes_generator = grid.polygonize(catch.astype(np.int32))

        # Convert to list and find the catchment polygon (value = 1)
        polygons = []
        for geom, value in shapes_generator:
            if value == 1:  # Catchment area
                polygons.append(shape(geom))

        if not polygons:
            raise ValueError("No catchment polygon generated. The point may be invalid.")

        # If multiple polygons, take the largest one (main catchment)
        if len(polygons) > 1:
            catchment_polygon = max(polygons, key=lambda p: p.area)
        else:
            catchment_polygon = polygons[0]

        snapped_point = Point(x_snap, y_snap)
        return catchment_polygon, snapped_point, flow_acc_at_pour_point

    except Exception as e:
        raise ValueError(f"Failed to convert catchment to polygon: {e}")


def delineate_catchment(
    point: Point,
    dem_path: Union[str, Path],
    dirmap: Tuple[int, int, int, int, int, int, int, int] = (64, 128, 1, 2, 4, 8, 16, 32),
    apply_input_mask: bool = False,
    nodata_in: Optional[float] = None,
    nodata_out: Optional[float] = None,
    pit_fill: bool = True,
    pit_fill_epsilon: float = 0.0001,
    fill_depressions: bool = True,
    resolve_flats: bool = True,
    routing: str = "d8",
    flow_acc_threshold: int = 5000,
) -> Polygon:
    """
    Delineate a catchment area for a given point using a Digital Elevation Model (DEM).

    This function uses the pysheds library to perform watershed delineation based on
    terrain analysis. The workflow includes:
    1. Loading the DEM raster
    2. Filling pits
    3. Filling depressions (multi-cell sinks)
    4. Resolving flats
    5. Computing flow direction
    6. Computing flow accumulation
    7. Snapping the pour point to the nearest high-accumulation cell
    8. Delineating the catchment

    Args:
        point: A Shapely Point representing the pour point (outlet) in the catchment.
               Should be in the same coordinate system as the DEM.
        dem_path: Path to the Digital Elevation Model raster file (e.g., GeoTIFF).
        dirmap: Direction mapping for flow direction (default is for D8 routing).
                Format: (N, NE, E, SE, S, SW, W, NW)
        apply_input_mask: Whether to apply the DEM's nodata mask to operations.
        nodata_in: Value representing nodata in the input DEM. If None, will use
                   the raster's nodata value.
        nodata_out: Value to use for nodata in output arrays. If None, the
                    function uses NumPy-scalar nodata values internally to
                    avoid dtype conflicts under NumPy 2.x (NEP 50).
        pit_fill: Whether to fill pits in the DEM before flow analysis.
        pit_fill_epsilon: Small value to add when filling pits to ensure drainage.
        fill_depressions: Whether to fill multi-cell depressions after pit
                          filling.  Recommended: True (default).
        resolve_flats: Whether to resolve flat areas in the DEM.
        routing: Flow routing algorithm ('d8' or 'dinf'). Default is 'd8'.
        flow_acc_threshold: Threshold for snapping pour point to stream network.
                           Higher values = snap to larger streams.

    Returns:
        A Shapely Polygon representing the delineated catchment boundary.

    Raises:
        FileNotFoundError: If the DEM file doesn't exist.
        ValueError: If the point is outside the DEM extent or if catchment
                    delineation fails.

    Example:
        >>> from shapely.geometry import Point
        >>> point = Point(-3.5, 50.7)  # Longitude, Latitude
        >>> catchment = delineate_catchment(
        ...     point=point,
        ...     dem_path="path/to/dem.tif",
        ...     flow_acc_threshold=1000
        ... )
        >>> print(f"Catchment area: {catchment.area} square degrees")
    """
    polygon, _snapped, _acc = _delineate_catchment_core(
        point=point,
        dem_path=dem_path,
        dirmap=dirmap,
        apply_input_mask=apply_input_mask,
        nodata_in=nodata_in,
        nodata_out=nodata_out,
        pit_fill=pit_fill,
        pit_fill_epsilon=pit_fill_epsilon,
        fill_depressions=fill_depressions,
        resolve_flats=resolve_flats,
        routing=routing,
        flow_acc_threshold=flow_acc_threshold,
    )
    return polygon


def delineate_catchment_with_metadata(point: Point, dem_path: Union[str, Path], **kwargs) -> dict:
    """
    Delineate a catchment and return additional metadata about the catchment.

    This is a wrapper around the core delineation that also returns useful
    catchment statistics, including the snapped pour point.

    Args:
        point: A Shapely Point representing the pour point.
        dem_path: Path to the DEM raster file.
        **kwargs: Additional keyword arguments passed to delineate_catchment.

    Returns:
        A dictionary containing:
            - 'polygon': The catchment polygon
            - 'area': Catchment area (in the units of the CRS)
            - 'centroid': Centroid of the catchment
            - 'bounds': Bounding box of the catchment
            - 'pour_point': The original (unsnapped) pour point
            - 'snapped_pour_point': The pour point after snapping to the
              stream network (where the catchment actually drains to)

    Example:
        >>> result = delineate_catchment_with_metadata(point, "dem.tif")
        >>> print(f"Catchment area: {result['area']:.2f} km²")
        >>> print(f"Centroid: {result['centroid']}")
        >>> print(f"Snapped pour point: {result['snapped_pour_point']}")
    """
    # Delineate catchment using the core function to get the snapped point
    catchment, snapped_point, flow_acc_at_pour_point = _delineate_catchment_core(
        point, dem_path, **kwargs
    )

    # Calculate metadata
    metadata = {
        "polygon": catchment,
        "area": catchment.area,
        "centroid": catchment.centroid,
        "bounds": catchment.bounds,
        "pour_point": point,
        "snapped_pour_point": snapped_point,
        "flow_acc_at_pour_point": flow_acc_at_pour_point,
    }

    return metadata


# ---------------------------------------------------------------------------
# Standalone flow-accumulation computation (for raster overlays / analysis)
# ---------------------------------------------------------------------------


def compute_flow_accumulation(
    dem_path: Union[str, Path],
    dirmap: Tuple[int, int, int, int, int, int, int, int] = (64, 128, 1, 2, 4, 8, 16, 32),
    routing: str = "d8",
    pit_fill: bool = True,
    pit_fill_epsilon: float = 0.0001,
    fill_depressions: bool = True,
    resolve_flats: bool = True,
    apply_input_mask: bool = False,
) -> Tuple[np.ndarray, Any, Any]:
    """Compute flow accumulation for an entire DEM raster.

    Runs the standard pysheds hydrological conditioning pipeline
    (pit-fill → resolve flats → flow direction → accumulation) on the
    supplied DEM and returns the resulting accumulation grid together with
    the rasterio ``Affine`` transform and CRS so the array can be
    reprojected for display.

    Parameters
    ----------
    dem_path : str or Path
        Path to a GeoTIFF DEM.
    dirmap : tuple
        D8 direction mapping.  Default is the pysheds convention
        ``(N, NE, E, SE, S, SW, W, NW) = (64, 128, 1, 2, 4, 8, 16, 32)``.
    routing : str
        Flow-routing algorithm.  Only ``"d8"`` is currently supported.
    pit_fill : bool
        Whether to fill pits before computing flow direction.
    pit_fill_epsilon : float
        Epsilon used when resolving flats.
    resolve_flats : bool
        Whether to resolve flat areas.
    apply_input_mask : bool
        Whether to apply the DEM nodata mask to the accumulation step.

    Returns
    -------
    acc_array : numpy.ndarray, shape (H, W), dtype float64
        Flow-accumulation values.  Cells with no valid data are ``nan``.
        Valid cells hold the number of upstream cells (≥ 1).
    transform : rasterio.transform.Affine
        Affine transform mapping pixel coordinates to the DEM's native CRS.
    crs : rasterio.crs.CRS
        Coordinate reference system of the DEM.

    Raises
    ------
    FileNotFoundError
        If *dem_path* does not exist.
    ValueError
        If an unsupported *routing* algorithm is requested.
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    logger.info("Computing flow accumulation for %s …", dem_path.name)

    # Read CRS and transform from rasterio (pysheds doesn't expose CRS).
    with rasterio.open(str(dem_path)) as src:
        crs = src.crs

    # Hydrological conditioning via pysheds
    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))

    # Always use explicit NumPy scalar nodata_out to avoid NEP-50 TypeError.
    _int_nodata_out: dict = {"nodata_out": np.int64(0)}

    if pit_fill:
        pit_filled_dem = grid.fill_pits(dem)
        if fill_depressions:
            flooded_dem = grid.fill_depressions(pit_filled_dem)
        else:
            flooded_dem = pit_filled_dem
        if resolve_flats:
            inflated_dem = grid.resolve_flats(flooded_dem, eps=pit_fill_epsilon)
        else:
            inflated_dem = flooded_dem
    else:
        inflated_dem = dem

    if routing.lower() != "d8":
        raise ValueError(f"Unsupported routing method: {routing!r}. Only 'd8' is supported.")

    fdir = grid.flowdir(inflated_dem, dirmap=dirmap, routing=routing, **_int_nodata_out)
    acc = grid.accumulation(
        fdir,
        dirmap=dirmap,
        routing=routing,
        apply_input_mask=apply_input_mask,
        **_int_nodata_out,
    )

    # Convert to float64 and mask nodata (0 when _int_nodata_out was used,
    # otherwise the raster nodata – accumulation is always ≥ 1 for valid cells).
    acc_array = np.array(acc, dtype=np.float64)
    acc_array[acc_array == 0] = np.nan

    transform = acc.affine
    logger.info(
        "Flow accumulation computed: shape=%s, max=%.0f",
        acc_array.shape,
        float(np.nanmax(acc_array)) if not np.all(np.isnan(acc_array)) else 0,
    )
    return acc_array, transform, crs
