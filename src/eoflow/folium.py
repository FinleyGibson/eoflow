import folium
from shapely.geometry import MultiPolygon, Polygon


def add_polygon_to_folium(
    polygon: Polygon | MultiPolygon,
    f_map=None,
    color="blue",
    weight=3,
    fill=True,
    fill_opacity=0.4,
    map_kwargs={},
):
    """
    Add a Shapely Polygon (or MultiPolygon) to a Folium map.

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
        exterior_coords = [
            (y, x) for x, y in polygon.exterior.coords
        ]  # folium uses (lat, lon)

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
