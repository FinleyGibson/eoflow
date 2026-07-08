"""
River network utilities for fetching OpenStreetMap waterway data via Overpass API.

This module provides functions to query the Overpass API for river/waterway data
and build topological network graphs for analysis.

Main Functions
--------------
get_river_network_from_shape(shp, ...)
    Query Overpass API to retrieve river/stream/canal features within a boundary.
    Returns a GeoDataFrame with LineString geometries.

build_river_network_graph(gdf, ...)
    Build a topological graph from LineString geometries where nodes are placed
    at intersections and endpoints, with edges representing river segments.

calculate_graph_length_meters(G, crs)
    Calculate total length of a river network graph in meters using Haversine
    formula for geographic coordinates.

Key Features
------------
- Accepts Shapely Polygon/MultiPolygon or GeoJSON-like mapping as boundary
- Queries Overpass with automatic retries and fallback endpoints (handles 504/5xx errors)
- Returns GeoPandas GeoDataFrame of LineString/MultiLineString geometries (EPSG:4326)
- Builds topological graphs with nodes at intersections and endpoints
- Calculates edge lengths in meters for network analysis

Usage Examples
--------------

Fetching river network data::

    from shapely.geometry import box
    from eoflow.rivers import get_river_network_from_shape

    # Define area of interest
    bbox = box(-0.12, 51.50, -0.10, 51.52)  # London area

    # Fetch river network (returns GeoDataFrame)
    rivers_gdf = get_river_network_from_shape(bbox, use_bbox=True)
    print(f"Found {len(rivers_gdf)} waterway features")

Building a topological network graph::

    from eoflow.rivers import (
        get_river_network_from_shape,
        build_river_network_graph,
        calculate_graph_length_meters
    )

    # Fetch river data
    bbox = box(-0.12, 51.50, -0.10, 51.52)
    gdf = get_river_network_from_shape(bbox, use_bbox=True)

    # Build topological graph (nodes at intersections/endpoints)
    G = build_river_network_graph(gdf)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Calculate total length in meters
    length_m = calculate_graph_length_meters(G, crs=gdf.crs)
    print(f"Total river length: {length_m/1000:.2f} km")

    # Analyze network topology
    import networkx as nx

    # Find confluences (where 3+ rivers meet)
    confluences = [n for n, d in G.degree() if d >= 3]
    print(f"Found {len(confluences)} confluences")

    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Network has {len(components)} connected components")

Error handling::

    from eoflow.rivers import get_river_network_from_shape, OverpassError

    try:
        gdf = get_river_network_from_shape(bbox, max_retries=8, use_bbox=True)
    except OverpassError as e:
        print(f"Overpass API error: {e}")
        # Try again later or use smaller area
    except ValueError as e:
        print(f"Invalid input: {e}")

Notes
-----
- The function performs HTTP requests to public Overpass endpoints which may rate-limit
  or return intermittent 502/504 errors; automatic retry with exponential backoff is implemented
- Use `use_bbox=True` for more reliable queries with rectangular areas
- For production use, consider adding caching (e.g., diskcache) to avoid repeated queries
- The topological graph is different from the OSM node-level graph - it creates nodes
  only at meaningful locations (intersections, endpoints, confluences)
- Edge lengths can be calculated in meters using the calculate_graph_length_meters function
"""

from __future__ import annotations

import time
from random import uniform
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.geometry import shape as shapely_shape
from shapely.ops import linemerge, substring

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    # NOTE: "https://overpass.osm.ch/api/interpreter" was previously listed
    # here as a fallback, but it responds 200 OK with an *empty* result set
    # for queries that unambiguously have matching data (verified against a
    # known river bbox). Because that looks like a legitimate success to
    # `_fetch_overpass`, it silently masked real data whenever the primary
    # endpoint hit a transient error and retries rotated onto it. Prefer
    # relying on retries against the authoritative overpass-api.de instance
    # instead of resurrecting that mirror as a fallback.
]
WATERWAY_VALUES = "river|stream|canal|drain|riverbank|ditch|brook"

# Overpass instances (notably overpass-api.de) reject requests that don't send
# a descriptive User-Agent, responding with "406 Not Acceptable" instead of
# serving the query. Always identify ourselves to avoid this.
_REQUEST_HEADERS = {
    "User-Agent": "eoflow-rivers/1.0 (+https://github.com/eoflow; contact: eoflow@example.com)"
}


logger = get_logger(__name__)


from requests.exceptions import ConnectionError, RequestException, Timeout


class OverpassError(Exception):
    pass


def _make_overpass_poly(geom: Union[Polygon, MultiPolygon]) -> str:
    """
    Convert a Shapely Polygon or MultiPolygon to an Overpass 'poly' string.
    Overpass expects "lat lon" pairs separated by spaces. Shapely uses (lon, lat).
    """
    polys: List[str] = []

    if isinstance(geom, Polygon):
        polygon_list = [geom]
    else:
        polygon_list = list(geom.geoms)

    for p in polygon_list:
        coords = list(p.exterior.coords)
        if len(coords) < 3:
            logger.warning("Polygon has fewer than 3 coordinates, skipping")
            continue
        # Overpass wants "lat lon"
        pair_strs = [f"{y:.6f} {x:.6f}" for (x, y) in coords]
        polys.append(" ".join(pair_strs))

    if not polys:
        raise ValueError("No valid polygons to convert to Overpass poly string")

    # If multiple polygons, Overpass accepts multiple poly lines separated by " ; "
    return " ; ".join(polys)


def _build_overpass_query_from_poly_str(
    poly_str: str, waterway_values: str = "river|stream|canal|drain|riverbank"
) -> str:
    """
    Build an Overpass QL query string that finds ways and relations with waterway tag values
    overlapping the provided poly area and also requests referenced nodes.
    """
    # Validate poly_str to avoid injection or malformed queries
    if not poly_str or len(poly_str.strip()) == 0:
        raise ValueError("poly_str cannot be empty")

    # Build query with timeout setting
    query = f"""[out:json][timeout:180];
(
  way["waterway"~"{waterway_values}"](poly:"{poly_str}");
  relation["waterway"~"{waterway_values}"](poly:"{poly_str}");
);
(._;>;);
out body;"""

    logger.debug("Built Overpass query with poly containing %d characters", len(poly_str))
    return query


def _build_overpass_bbox_query(
    bbox: tuple, waterway_values: str = "river|stream|canal|drain|riverbank"
) -> str:
    """
    Build an Overpass QL query using bounding box instead of polygon.
    More robust for simple rectangular areas.

    bbox: (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Build query with bbox selector
    query = f"""[out:json][timeout:180];
(
  way["waterway"~"{waterway_values}"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["waterway"~"{waterway_values}"]({min_lat},{min_lon},{max_lat},{max_lon});
);
(._;>;);
out body;"""

    logger.debug("Built Overpass bbox query for bounds: %s", bbox)
    return query


def _fetch_overpass(
    query: str,
    endpoints: Iterable[str],
    timeout: int = 180,
    max_retries: int = 6,
    backoff_factor: float = 1.5,
) -> Dict[str, Any]:

    endpoint_list = list(endpoints)
    if not endpoint_list:
        raise ValueError("At least one Overpass endpoint must be provided")

    last_exc: Optional[Exception] = None
    n_endpoints = len(endpoint_list)

    for attempt in range(max_retries):
        endpoint = endpoint_list[attempt % n_endpoints]

        try:
            logger.debug(
                "Overpass request attempt=%s/%s endpoint=%s",
                attempt + 1,
                max_retries,
                endpoint,
            )

            resp = requests.post(
                endpoint,
                data={"data": query},
                timeout=timeout,
                headers=_REQUEST_HEADERS,
            )

            # Handle HTTP-level errors explicitly
            if resp.status_code == 400:
                # Bad query → do NOT retry
                raise OverpassError(f"Bad Overpass query (400). Response: {resp.text[:500]}")

            if resp.status_code in (429, 500, 502, 503, 504):
                # Retryable server-side errors
                raise OverpassError(f"Retryable HTTP error {resp.status_code}")

            resp.raise_for_status()

            try:
                return resp.json()
            except ValueError as exc:
                # JSON decode failed → likely Overpass HTML error or empty response
                raise OverpassError(
                    f"Invalid JSON response from Overpass. First 300 chars:\n{resp.text[:300]}"
                ) from exc

        except (Timeout, ConnectionError) as exc:
            last_exc = exc
            logger.warning(
                "Network error from %s: %s (attempt %s/%s)",
                endpoint,
                exc,
                attempt + 1,
                max_retries,
            )

        except OverpassError as exc:
            last_exc = exc
            logger.warning(
                "Overpass error: %s (attempt %s/%s)",
                exc,
                attempt + 1,
                max_retries,
            )

        except RequestException as exc:
            last_exc = exc
            logger.exception("Unexpected requests error: %s", exc)

        # Backoff + jitter
        sleep_time = (backoff_factor**attempt) + uniform(0, 0.3)
        time.sleep(sleep_time)

    raise OverpassError(
        f"Failed to contact Overpass after {max_retries} attempts. Last error: {last_exc}"
    )


# def _fetch_overpass(
#     query: str,
#     endpoints: Iterable[str],
#     timeout: int = 180,
#     max_retries: int = 6,
#     backoff_factor: float = 1.5,
# ) -> Dict[str, Any]:
#     """
#     Execute the Overpass QL `query` against one of the provided `endpoints`,
#     rotating and retrying on transient failures.

#     Returns the parsed JSON response on success or raises OverpassError.
#     """
#     endpoint_list = list(endpoints)
#     if not endpoint_list:
#         raise ValueError("At least one Overpass endpoint must be provided")

#     last_exc: Optional[Exception] = None
#     attempt = 0
#     n_endpoints = len(endpoint_list)

#     while attempt < max_retries:
#         endpoint = endpoint_list[attempt % n_endpoints]
#         try:
#             logger.debug(
#                 "Posting Overpass query attempt=%s/%s endpoint=%s",
#                 attempt + 1,
#                 max_retries,
#                 endpoint,
#             )
#             resp = requests.post(endpoint, data={"data": query}, timeout=timeout)
#             data = resp.json()
#             return data

#         except (requests.Timeout, requests.ConnectionError) as exc:
#             last_exc = exc
#             logger.warning(
#                 "Overpass connection/timeout error from %s: %s (attempt %s/%s)",
#                 endpoint,
#                 exc,
#                 attempt + 1,
#                 max_retries,
#             )
#             time.sleep(backoff_factor**attempt)
#             attempt += 1
#             continue
#         except OverpassError:
#             # Re-raise our own errors immediately
#             raise
#         except Exception as exc:
#             # Unexpected exception - capture and retry a limited number of times
#             last_exc = exc
#             logger.exception("Unexpected error contacting Overpass at %s: %s", endpoint, exc)
#             time.sleep(backoff_factor**attempt)
#             attempt += 1
#             continue

#     raise OverpassError(
#         f"Failed to contact Overpass after {max_retries} attempts. Last error: {last_exc}"
#     )


def _aggregate_osm_elements(
    elements: List[Dict[str, Any]],
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Parse Overpass API response and aggregate elements into nodes, ways, and relations.

    Parameters
    ----------
    elements : List[Dict[str, Any]]
        Raw elements from Overpass API JSON response

    Returns
    -------
    tuple
        (nodes_dict, ways_dict, relations_dict) where:
        - nodes_dict: {node_id: (lon, lat)}
        - ways_dict: {way_id: way_data_dict}
        - relations_dict: {relation_id: relation_data_dict}
    """
    nodes: Dict[int, Tuple[float, float]] = {}
    ways: Dict[int, Dict[str, Any]] = {}
    relations: Dict[int, Dict[str, Any]] = {}

    for el in elements:
        el_type = el.get("type")
        if el_type == "node":
            if "lon" in el and "lat" in el:
                nodes[el["id"]] = (el["lon"], el["lat"])
        elif el_type == "way":
            ways[el["id"]] = el
        elif el_type == "relation":
            relations[el["id"]] = el

    return nodes, ways, relations


def _convert_ways_to_rows(
    ways: Dict[int, Dict[str, Any]], nodes: Dict[int, Tuple[float, float]]
) -> List[Dict[str, Any]]:
    """
    Convert OSM ways into GeoDataFrame rows with LineString geometries.

    Parameters
    ----------
    ways : Dict[int, Dict[str, Any]]
        Mapping of way IDs to way data from Overpass API
    nodes : Dict[int, Tuple[float, float]]
        Mapping of node IDs to (lon, lat) coordinates

    Returns
    -------
    List[Dict[str, Any]]
        List of row dictionaries with osm_id, osm_type, geometry, and tags
    """
    rows: List[Dict[str, Any]] = []

    for way_id, w in ways.items():
        node_refs = w.get("nodes", [])
        coords: List[Tuple[float, float]] = []
        missing = False

        for nid in node_refs:
            if nid in nodes:
                coords.append(nodes[nid])
            else:
                missing = True
                break

        if missing or len(coords) < 2:
            continue

        geom_line = LineString(coords)
        rows.append(
            {
                "osm_id": way_id,
                "osm_type": "way",
                "geometry": geom_line,
                "tags": w.get("tags", {}),
            }
        )

    return rows


def _convert_relations_to_rows(
    relations: Dict[int, Dict[str, Any]],
    ways: Dict[int, Dict[str, Any]],
    nodes: Dict[int, Tuple[float, float]],
    simplify_multiline: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert OSM relations (multipart features) into GeoDataFrame rows.

    Stitches together member ways and creates merged LineString or MultiLineString geometries.

    Parameters
    ----------
    relations : Dict[int, Dict[str, Any]]
        Mapping of relation IDs to relation data from Overpass API
    ways : Dict[int, Dict[str, Any]]
        Mapping of way IDs to way data (for member way lookup)
    nodes : Dict[int, Tuple[float, float]]
        Mapping of node IDs to (lon, lat) coordinates
    simplify_multiline : bool, optional
        If True, merge contiguous segments into single LineString where possible.
        Default is True.

    Returns
    -------
    List[Dict[str, Any]]
        List of row dictionaries with osm_id, osm_type, geometry, and tags
    """
    rows: List[Dict[str, Any]] = []

    for rel_id, r in relations.items():
        member_ways = [m for m in r.get("members", []) if m.get("type") == "way"]
        lines: List[LineString] = []

        for m in member_ways:
            ref = m.get("ref")
            w = ways.get(ref)
            if not w:
                continue

            node_refs = w.get("nodes", [])
            coords = [nodes[nid] for nid in node_refs if nid in nodes]

            if len(coords) >= 2:
                lines.append(LineString(coords))

        if not lines:
            continue

        if simplify_multiline:
            merged = linemerge(lines)
        else:
            merged = MultiLineString(lines) if len(lines) > 1 else lines[0]

        rows.append(
            {
                "osm_id": rel_id,
                "osm_type": "relation",
                "geometry": merged,
                "tags": r.get("tags", {}),
            }
        )

    return rows


def _build_osm_node_graph(
    gdf: pd.DataFrame,
    nodes: Dict[int, Tuple[float, float]],
    ways: Dict[int, Dict[str, Any]],
    geom: Union[Polygon, MultiPolygon],
) -> nx.Graph:
    """
    Build a NetworkX graph from OSM ways using node-level connectivity.

    Creates edges between consecutive nodes in each way, representing the OSM data structure
    (different from the topological graph created by build_river_network_graph).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the processed river features
    nodes : Dict[int, Tuple[float, float]]
        Mapping of node IDs to (lon, lat) coordinates
    ways : Dict[int, Dict[str, Any]]
        Mapping of way IDs to way data
    geom : Union[Polygon, MultiPolygon]
        Boundary geometry for filtering edges

    Returns
    -------
    nx.Graph
        NetworkX graph with nodes representing OSM nodes and edges between consecutive node pairs.
        Edges have attributes: way_id, tags, length
    """
    G = nx.Graph()

    way_ids_in_gdf: set[str] = set(gdf["osm_id"].loc[gdf["osm_type"] == "way"].values)

    for way_id, w in ways.items():
        # Skip ways that were filtered out during clipping
        if way_id not in way_ids_in_gdf:
            continue

        node_refs = w.get("nodes", [])
        for idx in range(len(node_refs) - 1):
            u = node_refs[idx]
            v = node_refs[idx + 1]

            if u not in nodes or v not in nodes:
                continue

            ux, uy = nodes[u]
            vx, vy = nodes[v]

            # Check if this edge segment is within the boundary
            edge_geom = LineString([(ux, uy), (vx, vy)])
            if not edge_geom.intersects(geom):
                continue

            if not G.has_node(u):
                G.add_node(u, x=ux, y=uy)
            if not G.has_node(v):
                G.add_node(v, x=vx, y=vy)

            if G.has_edge(u, v):
                ways_list = G[u][v].get("ways", [])
                if way_id not in ways_list:
                    ways_list.append(way_id)
                G[u][v]["ways"] = ways_list
            else:
                seg_length = edge_geom.length
                G.add_edge(u, v, way_id=way_id, tags=w.get("tags", {}), length=seg_length)

    return G


def get_river_network_from_poly(
    shp: Union[dict, Polygon, MultiPolygon],
    overpass_endpoints: Iterable[str] = OVERPASS_ENDPOINTS,
    timeout: int = 180,
    waterway_values: str = WATERWAY_VALUES,
    return_graph: bool = False,
    simplify_multiline: bool = True,
    max_retries: int = 6,
    backoff_factor: float = 1.5,
    use_bbox: bool = False,
) -> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, nx.Graph]]:
    """
    Query Overpass to get river/stream ways and relations inside/overlapping the provided shape.

    Parameters
    - shp: a Shapely Polygon/MultiPolygon or a GeoJSON-like dict (will be converted via shapely.shape).
    - overpass_endpoints: list/iterable of Overpass API endpoints to try. If None, a default list is used.
    - waterway_values: a string of OSM waterway values to filter by (default: "river|stream|canal|drain|riverbank|ditch|brook").
    - timeout: request timeout in seconds for each HTTP request.
    - return_graph: if True also return a NetworkX graph (nodes with 'x','y' attrs and edges with way id, tags).
    - simplify_multiline: if True, merge contiguous segments of a relation into one LineString where possible.
                         If False, relations with multiple member ways will create MultiLineString geometries.
    - max_retries: maximum number of attempts across endpoints.
    - backoff_factor: exponential backoff base used between retries.
    - use_bbox: if True, use bounding box query instead of polygon query (more robust but less precise).

    Returns:
    - GeoDataFrame with columns: ['osm_id', 'osm_type', 'geometry', 'tags'] and any additional attributes from OSM.
    - If return_graph=True returns (gdf, graph).

    Raises OverpassError on failure to get a valid Overpass response.
    """
    # Normalize input geometry
    if isinstance(shp, dict):
        geom = shapely_shape(shp)
    else:
        geom = shp

    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise ValueError(
            "The provided shape must be a Polygon or MultiPolygon (or GeoJSON mapping)."
        )

    # Choose query method: bbox is more robust, poly is more precise
    if use_bbox:
        # Use bounding box query
        bounds = geom.bounds  # (min_x, min_y, max_x, max_y)
        query = _build_overpass_bbox_query(bounds, waterway_values=waterway_values)
        logger.info("Querying Overpass for waterways in bbox: %s", bounds)
    else:
        # Use polygon query (more precise but can fail with large/complex polygons)
        try:
            poly_str = _make_overpass_poly(geom)
        except ValueError as e:
            raise ValueError(f"Failed to convert shape to Overpass poly format: {e}")

        try:
            query = _build_overpass_query_from_poly_str(poly_str, waterway_values=waterway_values)
        except ValueError as e:
            raise ValueError(f"Failed to build Overpass query: {e}")

        logger.info(
            "Querying Overpass for waterways in boundary (poly string: %d chars)",
            len(poly_str),
        )

    # Fetch data
    data = _fetch_overpass(
        query,
        overpass_endpoints,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
    )
    elements = data.get("elements", [])

    # Aggregate elements into nodes, ways, and relations
    nodes, ways, relations = _aggregate_osm_elements(elements)

    # Convert OSM features to GeoDataFrame rows
    rows: List[Dict[str, Any]] = []
    rows.extend(_convert_ways_to_rows(ways, nodes))
    rows.extend(
        _convert_relations_to_rows(relations, ways, nodes, simplify_multiline=simplify_multiline)
    )

    # Build GeoDataFrame
    if not rows:
        # build empty gpd and graph if no rivers are found
        gdf = gpd.GeoDataFrame(
            {
                "osm_id": [],
                "osm_type": [],
                "tags": [],
                "geometry": [],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
    else:
        gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(
            pd.DataFrame(rows), geometry="geometry", crs="EPSG:4326"
        )

        # Clip to the provided polygon to ensure features are bounded by input
        try:
            gdf["geometry"] = gdf["geometry"].intersection(geom)
        except Exception:
            logger.exception(
                "Failed to intersect geometries with provided shape;       returning un-clipped results."
            )

        gdf = gpd.GeoDataFrame(gdf.reset_index(drop=True))

    if return_graph:
        nxg = (
            _build_osm_node_graph(gdf, nodes, ways, geom) if not (gdf.shape[0] == 0) else nx.Graph()
        )
        return gdf, nxg
    else:
        return gdf


def calculate_graph_length_meters(G: nx.Graph, crs: Any = "EPSG:4326") -> float:
    """
    Calculate total length of a river network graph in meters.

    If the graph edge lengths are in degrees (geographic CRS), this function
    will estimate lengths in meters using the Haversine formula.

    Parameters
    ----------
    G : nx.Graph
        Graph with 'length' attribute on edges
    crs : Any, optional
        Coordinate reference system of the graph node coordinates.
        If geographic (e.g., EPSG:4326), uses Haversine formula.
        Default is "EPSG:4326".

    Returns
    -------
    float
        Total length in meters

    Examples
    --------
    >>> from eoflow.rivers import get_river_network_from_shape, build_river_network_graph
    >>> from shapely.geometry import box
    >>> bbox = box(-0.12, 51.50, -0.10, 51.52)
    >>> gdf = get_river_network_from_shape(bbox, use_bbox=True)
    >>> G = build_river_network_graph(gdf)
    >>> length_m = calculate_graph_length_meters(G)
    >>> print(f"Total river length: {length_m:.2f} meters")
    """
    from math import atan2, cos, radians, sin, sqrt

    # Check if CRS is geographic
    is_geographic = False
    if isinstance(crs, str):
        is_geographic = "4326" in crs or "WGS" in crs.upper()
    elif hasattr(crs, "is_geographic"):
        is_geographic = crs.is_geographic

    total_length = 0.0

    if is_geographic:
        # Use Haversine formula for geographic coordinates
        R = 6371000  # Earth radius in meters

        for u, v, data in G.edges(data=True):
            if "length" in data:
                # Get node coordinates
                x1, y1 = G.nodes[u].get("x", 0), G.nodes[u].get("y", 0)
                x2, y2 = G.nodes[v].get("x", 0), G.nodes[v].get("y", 0)

                # Haversine formula
                lat1, lon1 = radians(y1), radians(x1)
                lat2, lon2 = radians(y2), radians(x2)

                d_lat = lat2 - lat1
                d_lon = lon2 - lon1

                a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                length_m = R * c

                total_length += length_m
    else:
        # Assume lengths are already in meters or similar linear units
        total_length = sum(data.get("length", 0) for _, _, data in G.edges(data=True))

    return total_length


def build_river_network_graph(
    gdf: gpd.GeoDataFrame,
    length_col: str = "length",
    tolerance: float = 1e-8,
) -> nx.Graph:
    """
    Build a topological graph from a GeoDataFrame of river LineString geometries.

    Creates nodes at line intersections and endpoints, with edges representing
    the river segments between nodes. This is useful for topological analysis
    of river networks.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with LineString or MultiLineString geometries (e.g., from get_river_network_from_shape).
    length_col : str, default="length"
        Name of the column to store edge lengths in the graph.
    tolerance : float, default=1e-8
        Coordinate tolerance for identifying identical points (in the GeoDataFrame's CRS units).

    Returns
    -------
    nx.Graph
        NetworkX graph where:
        - Nodes have 'x', 'y' attributes (coordinates)
        - Edges have 'length', 'geometry' (LineString), and 'source_idx' (original GeoDataFrame index) attributes

    Examples
    --------
    >>> from shapely.geometry import box
    >>> from eoflow.rivers import get_river_network_from_shape, build_river_network_graph
    >>>
    >>> # Get river network
    >>> bbox = box(-0.12, 51.50, -0.10, 51.52)
    >>> gdf = get_river_network_from_shape(bbox, use_bbox=True)
    >>>
    >>> # Build topological graph
    >>> G = build_river_network_graph(gdf)
    >>> print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    >>>
    >>> # Analyze network
    >>> import networkx as nx
    >>> # Find connected components
    >>> components = list(nx.connected_components(G))
    >>> print(f"Network has {len(components)} connected components")
    >>>
    >>> # Calculate total network length
    >>> total_length = sum(G[u][v]['length'] for u, v in G.edges())
    >>> print(f"Total network length: {total_length:.2f} units")

    Notes
    -----
    - MultiLineString geometries are exploded into individual LineStrings
    - Lines are split at intersection points and endpoints
    - Coordinate precision is important - use `tolerance` to handle floating point errors
    - For geographic coordinates (EPSG:4326), consider reprojecting to a projected CRS
      for more accurate length calculations in meters
    """

    # Explode MultiLineStrings into individual LineStrings
    exploded = gdf.explode(index_parts=False).reset_index(drop=True)

    # Filter to only LineString geometries
    line_gdf = exploded[exploded.geometry.type == "LineString"].copy()

    if len(line_gdf) == 0:
        logger.warning("No LineString geometries found in GeoDataFrame")
        return nx.Graph()

    logger.info(f"Processing {len(line_gdf)} LineStrings to build network graph")

    # Step 1: Find all potential node locations (endpoints and intersections)
    node_coords = set()

    # Add all endpoints
    for idx, row in line_gdf.iterrows():
        geom = row.geometry
        if geom and not geom.is_empty:
            coords = list(geom.coords)
            if len(coords) >= 2:
                # Add start and end points
                node_coords.add(coords[0])
                node_coords.add(coords[-1])

    # Step 2: Find intersections between lines
    # This is computationally expensive for large datasets, so we use spatial index
    s_index = line_gdf.sindex

    for idx, row in line_gdf.iterrows():
        geom = row.geometry
        if not geom or geom.is_empty:
            continue

        # Find candidate intersecting lines using spatial index
        possible_matches_idx = list(s_index.intersection(geom.bounds))
        possible_matches = line_gdf.iloc[possible_matches_idx]

        for idx2, row2 in possible_matches.iterrows():
            if idx >= idx2:  # Avoid duplicate checks
                continue

            geom2 = row2.geometry
            if not geom2 or geom2.is_empty:
                continue

            # Check if lines intersect
            if geom.intersects(geom2):
                intersection = geom.intersection(geom2)

                # Handle different intersection types
                if intersection.geom_type == "Point":
                    node_coords.add((intersection.x, intersection.y))
                elif intersection.geom_type == "MultiPoint":
                    for pt in intersection.geoms:
                        node_coords.add((pt.x, pt.y))
                elif intersection.geom_type == "LineString":
                    # Lines overlap - add endpoints of overlap
                    coords = list(intersection.coords)
                    if coords:
                        node_coords.add(coords[0])
                        node_coords.add(coords[-1])
                elif intersection.geom_type == "MultiLineString":
                    for line in intersection.geoms:
                        coords = list(line.coords)
                        if coords:
                            node_coords.add(coords[0])
                            node_coords.add(coords[-1])

    # Step 3: Create node mapping with tolerance-based deduplication
    # Group nearby coordinates together
    node_list = sorted(node_coords)
    node_id_map = {}  # Maps coordinate tuple to node ID
    nodes_data = {}  # Maps node ID to coordinate
    next_node_id = 0

    for coord in node_list:
        # Check if this coordinate is close to any existing node
        found = False
        for existing_coord, node_id in node_id_map.items():
            dx = abs(coord[0] - existing_coord[0])
            dy = abs(coord[1] - existing_coord[1])
            if dx < tolerance and dy < tolerance:
                node_id_map[coord] = node_id
                found = True
                break

        if not found:
            node_id_map[coord] = next_node_id
            nodes_data[next_node_id] = coord
            next_node_id += 1

    logger.info(f"Identified {len(nodes_data)} unique node locations")

    # Step 4: Build graph by splitting lines at node locations
    G = nx.Graph()

    # Add nodes with coordinates
    for node_id, (x, y) in nodes_data.items():
        G.add_node(node_id, x=x, y=y)

    # Helper function to find closest node to a coordinate
    def find_node_id(coord):
        for existing_coord, node_id in node_id_map.items():
            dx = abs(coord[0] - existing_coord[0])
            dy = abs(coord[1] - existing_coord[1])
            if dx < tolerance and dy < tolerance:
                return node_id
        # If not found, create new node (shouldn't happen but handle gracefully)
        nonlocal next_node_id
        node_id = next_node_id
        next_node_id += 1
        node_id_map[coord] = node_id
        nodes_data[node_id] = coord
        G.add_node(node_id, x=coord[0], y=coord[1])
        return node_id

    # Process each line and split at nodes.
    #
    # Rather than only splitting where a node coordinate happens to already
    # be a vertex of the line's coordinate sequence, project every known
    # node onto the line and split at its along-line distance. This
    # correctly handles intersections that occur strictly between two
    # vertices (e.g. two lines crossing in an "X" shape), not just at
    # shared vertices.
    for orig_idx, row in line_gdf.iterrows():
        geom = row.geometry
        if not geom or geom.is_empty:
            continue

        line_length = geom.length
        if line_length == 0:
            continue

        start_node = find_node_id(geom.coords[0])
        end_node = find_node_id(geom.coords[-1])

        # Find every known node that lies on (or very near) this line, and
        # record how far along the line it sits.
        node_distance_along_line = {start_node: 0.0, end_node: line_length}
        for node_id, coord in nodes_data.items():
            if node_id in node_distance_along_line:
                continue
            pt = Point(coord)
            if geom.distance(pt) <= tolerance:
                node_distance_along_line[node_id] = geom.project(pt)

        # Order nodes by distance along the line, collapsing any that fall
        # within `tolerance` of the previous one (e.g. floating point noise
        # or coincident endpoints).
        ordered_nodes = sorted(node_distance_along_line.items(), key=lambda kv: kv[1])
        collapsed_nodes = [ordered_nodes[0]]
        for node_id, dist in ordered_nodes[1:]:
            if dist - collapsed_nodes[-1][1] <= tolerance:
                continue
            collapsed_nodes.append((node_id, dist))

        # Create edges between consecutive nodes along this line
        for i in range(len(collapsed_nodes) - 1):
            node1, dist1 = collapsed_nodes[i]
            node2, dist2 = collapsed_nodes[i + 1]

            if node1 == node2:
                continue

            segment_geom = substring(geom, dist1, dist2)
            if segment_geom.is_empty or segment_geom.length == 0:
                continue
            segment_length = segment_geom.length

            # Add edge (or update if already exists with shorter path)
            if G.has_edge(node1, node2):
                # Keep the shorter segment
                if segment_length < G[node1][node2].get(length_col, float("inf")):
                    G[node1][node2][length_col] = segment_length
                    G[node1][node2]["geometry"] = segment_geom
                    G[node1][node2]["source_idx"] = orig_idx
            else:
                G.add_edge(
                    node1,
                    node2,
                    **{
                        length_col: segment_length,
                        "geometry": segment_geom,
                        "source_idx": orig_idx,
                    },
                )

    logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G


def calculate_shortest_path_length(
    G: nx.Graph,
    source_node: int,
    target_node: int,
    crs: Any = "EPSG:4326",
) -> float:
    """
    Calculate the shortest path length between two nodes in a river network graph, in metres.

    Uses Dijkstra's algorithm to find the shortest path and sums the edge lengths
    along that path. If the graph edge lengths are in degrees (geographic CRS),
    this function estimates lengths in meters using the Haversine formula.

    Parameters
    ----------
    G : nx.Graph
        Graph with 'length' attribute on edges and node coordinates ('x', 'y' attributes).
    source_node : int
        Starting node ID
    target_node : int
        Ending node ID
    crs : Any, optional
        Coordinate reference system of the graph node coordinates.
        If geographic (e.g., EPSG:4326), uses Haversine formula to convert
        degree-based edge lengths to meters.
        Default is "EPSG:4326".

    Returns
    -------
    float
        Total shortest path length in meters

    Raises
    ------
    nx.NetworkXNoPath
        If no path exists between the source and target nodes
    nx.NodeNotFound
        If either source_node or target_node is not in the graph

    Examples
    --------
    >>> from eoflow.rivers import get_river_network_from_shape, build_river_network_graph
    >>> from shapely.geometry import box
    >>>
    >>> bbox = box(-0.12, 51.50, -0.10, 51.52)
    >>> gdf = get_river_network_from_shape(bbox, use_bbox=True)
    >>> G = build_river_network_graph(gdf)
    >>>
    >>> # Get two nodes from the graph
    >>> nodes = list(G.nodes())
    >>> if len(nodes) >= 2:
    ...     node1, node2 = nodes[0], nodes[1]
    ...     length_m = calculate_shortest_path_length(G, node1, node2)
    ...     print(f"Shortest path: {length_m:.2f} meters")
    """
    from math import atan2, cos, radians, sin, sqrt

    # Check if source and target nodes exist
    if source_node not in G:
        raise nx.NodeNotFound(f"Source node {source_node} not found in graph")
    if target_node not in G:
        raise nx.NodeNotFound(f"Target node {target_node} not found in graph")

    # Find shortest path using Dijkstra's algorithm
    try:
        path = nx.shortest_path(G, source_node, target_node, weight="length")
    except nx.NetworkXNoPath:
        raise nx.NetworkXNoPath(f"No path exists between nodes {source_node} and {target_node}")

    # Check if CRS is geographic
    is_geographic = False
    if isinstance(crs, str):
        is_geographic = "4326" in crs or "WGS" in crs.upper()
    elif hasattr(crs, "is_geographic"):
        is_geographic = crs.is_geographic

    total_length = 0.0

    if is_geographic:
        # Use Haversine formula for geographic coordinates
        R = 6371000  # Earth radius in meters

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            # Get node coordinates
            x1, y1 = G.nodes[u].get("x", 0), G.nodes[u].get("y", 0)
            x2, y2 = G.nodes[v].get("x", 0), G.nodes[v].get("y", 0)

            # Haversine formula
            lat1, lon1 = radians(y1), radians(x1)
            lat2, lon2 = radians(y2), radians(x2)

            d_lat = lat2 - lat1
            d_lon = lon2 - lon1

            a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            length_m = R * c

            total_length += length_m
    else:
        # Assume lengths are already in meters or similar linear units
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            if G.has_edge(u, v):
                total_length += G[u][v].get("length", 0)

    return total_length
