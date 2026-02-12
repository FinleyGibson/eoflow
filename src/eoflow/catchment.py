from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from pysheds.grid import Grid
from shapely.geometry import Point, Polygon, shape


def delineate_catchment(
    point: Point,
    dem_path: Union[str, Path],
    dirmap: Tuple[int, int, int, int, int, int, int, int] = (64, 128, 1, 2, 4, 8, 16, 32),
    apply_input_mask: bool = False,
    nodata_in: Optional[float] = None,
    nodata_out: Optional[float] = None,
    pit_fill: bool = True,
    pit_fill_epsilon: float = 0.0001,
    resolve_flats: bool = True,
    routing: str = "d8",
    flow_acc_threshold: int = 1000,
) -> Polygon:
    """
    Delineate a catchment area for a given point using a Digital Elevation Model (DEM).

    This function uses the pysheds library to perform watershed delineation based on
    terrain analysis. The workflow includes:
    1. Loading the DEM raster
    2. Filling pits and resolving flats
    3. Computing flow direction
    4. Computing flow accumulation
    5. Snapping the pour point to the nearest high-accumulation cell
    6. Delineating the catchment

    Args:
        point: A Shapely Point representing the pour point (outlet) in the catchment.
               Should be in the same coordinate system as the DEM.
        dem_path: Path to the Digital Elevation Model raster file (e.g., GeoTIFF).
        dirmap: Direction mapping for flow direction (default is for D8 routing).
                Format: (N, NE, E, SE, S, SW, W, NW)
        apply_input_mask: Whether to apply the DEM's nodata mask to operations.
        nodata_in: Value representing nodata in the input DEM. If None, will use
                   the raster's nodata value.
        nodata_out: Value to use for nodata in output arrays. If None, pysheds
                    manages nodata per step automatically (recommended to avoid
                    dtype conflicts with NumPy 2.x).
        pit_fill: Whether to fill pits in the DEM before flow analysis.
        pit_fill_epsilon: Small value to add when filling pits to ensure drainage.
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

    # Step 1: Fill pits in DEM
    if pit_fill:
        pit_filled_dem = grid.fill_pits(dem, **_nodata_in_kw, **_nodata_out_kw)

        # Step 2: Resolve flats
        if resolve_flats:
            inflated_dem = grid.resolve_flats(
                pit_filled_dem, eps=pit_fill_epsilon, **_nodata_out_kw
            )
        else:
            inflated_dem = pit_filled_dem
    else:
        inflated_dem = dem

    # Step 3: Compute flow direction
    if routing.lower() == "d8":
        fdir = grid.flowdir(
            inflated_dem,
            dirmap=dirmap,
            routing=routing,
        )
    else:
        raise ValueError(f"Unsupported routing method: {routing}. Use 'd8' or 'dinf'.")

    # Step 4: Compute flow accumulation
    acc = grid.accumulation(
        fdir,
        dirmap=dirmap,
        routing=routing,
        apply_input_mask=apply_input_mask,
    )

    # Step 5: Snap pour point to nearest high-accumulation cell
    # This ensures the pour point is on a stream rather than a hillslope
    try:
        x_snap, y_snap = grid.snap_to_mask(acc > flow_acc_threshold, (x, y), return_dist=False)
    except Exception as e:
        # If snapping fails, try using the original point
        print(f"Warning: Could not snap point to stream network: {e}")
        print(f"Using original point coordinates: ({x}, {y})")
        x_snap, y_snap = x, y

    # Step 6: Delineate the catchment
    try:
        catch = grid.catchment(
            x=x_snap,
            y=y_snap,
            fdir=fdir,
            dirmap=dirmap,
            routing=routing,
            xytype="coordinate",
        )
    except Exception as e:
        raise ValueError(f"Failed to delineate catchment: {e}")

    # Step 7: Convert catchment raster to polygon
    # Extract the catchment boundary as a polygon
    try:
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

        return catchment_polygon

    except Exception as e:
        raise ValueError(f"Failed to convert catchment to polygon: {e}")


def delineate_catchment_with_metadata(point: Point, dem_path: Union[str, Path], **kwargs) -> dict:
    """
    Delineate a catchment and return additional metadata about the catchment.

    This is a wrapper around delineate_catchment that also returns useful
    catchment statistics.

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
            - 'pour_point': The (possibly snapped) pour point

    Example:
        >>> result = delineate_catchment_with_metadata(point, "dem.tif")
        >>> print(f"Catchment area: {result['area']:.2f} km²")
        >>> print(f"Centroid: {result['centroid']}")
    """
    # Delineate catchment
    catchment = delineate_catchment(point, dem_path, **kwargs)

    # Calculate metadata
    metadata = {
        "polygon": catchment,
        "area": catchment.area,
        "centroid": catchment.centroid,
        "bounds": catchment.bounds,
        "pour_point": point,
    }

    return metadata
