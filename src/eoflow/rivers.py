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

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry import shape as shapely_shape
from shapely.ops import linemerge

logger = logging.getLogger(__name__)


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


def _build_overpass_query(poly_str: str, waterway_values: str = "river|stream|canal|drain|riverbank") -> str:
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


def _build_overpass_bbox_query(bbox: tuple, waterway_values: str = "river|stream|canal|drain|riverbank") -> str:
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


class OverpassError(RuntimeError):
    """Raised when Overpass API queries fail after retries."""


def _fetch_overpass(
    query: str,
    endpoints: Iterable[str],
    timeout: int = 180,
    max_retries: int = 6,
    backoff_factor: float = 1.5,
) -> Dict[str, Any]:
    """
    Execute the Overpass QL `query` against one of the provided `endpoints`,
    rotating and retrying on transient failures.

    Returns the parsed JSON response on success or raises OverpassError.
    """
    endpoint_list = list(endpoints)
    if not endpoint_list:
        raise ValueError("At least one Overpass endpoint must be provided")

    last_exc: Optional[Exception] = None
    attempt = 0
    n_endpoints = len(endpoint_list)

    while attempt < max_retries:
        endpoint = endpoint_list[attempt % n_endpoints]
        try:
            logger.debug("Posting Overpass query attempt=%s/%s endpoint=%s", attempt + 1, max_retries, endpoint)
            resp = requests.post(endpoint, data={"data": query}, timeout=timeout)
            # Raise for HTTP errors
            try:
                resp.raise_for_status()
            except requests.HTTPError as http_err:
                status = getattr(resp, "status_code", None)
                # Treat server errors as transient
                if status and 500 <= status < 600:
                    last_exc = http_err
                    logger.warning("Overpass server error %s from %s; retrying (attempt %s/%s)", status, endpoint, attempt + 1, max_retries)
                    time.sleep(backoff_factor ** attempt)
                    attempt += 1
                    continue
                # Client errors are non-retriable
                raise OverpassError(f"Overpass HTTP error from {endpoint}: {http_err}")

            # Try to parse JSON
            try:
                data = resp.json()
            except ValueError as exc:
                # Log response text for debugging
                resp_text = resp.text[:500] if resp.text else "(empty response)"
                logger.error("Failed to parse JSON from %s. Response: %s", endpoint, resp_text)
                # If response is empty or looks like HTML error, treat as transient
                if not resp.text or resp.text.strip().startswith("<"):
                    last_exc = exc
                    logger.warning("Empty or HTML response from %s, treating as transient; retrying (attempt %s/%s)", endpoint, attempt + 1, max_retries)
                    time.sleep(backoff_factor ** attempt)
                    attempt += 1
                    continue
                raise OverpassError(f"Invalid JSON from Overpass at {endpoint}: {exc}. Response: {resp_text}")

            return data

        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            logger.warning("Overpass connection/timeout error from %s: %s (attempt %s/%s)", endpoint, exc, attempt + 1, max_retries)
            time.sleep(backoff_factor ** attempt)
            attempt += 1
            continue
        except OverpassError:
            # Re-raise our own errors immediately
            raise
        except Exception as exc:
            # Unexpected exception - capture and retry a limited number of times
            last_exc = exc
            logger.exception("Unexpected error contacting Overpass at %s: %s", endpoint, exc)
            time.sleep(backoff_factor ** attempt)
            attempt += 1
            continue

    raise OverpassError(f"Failed to contact Overpass after {max_retries} attempts. Last error: {last_exc}")


def _aggregate_osm_elements(
    elements: List[Dict[str, Any]]
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
    ways: Dict[int, Dict[str, Any]],
    nodes: Dict[int, Tuple[float, float]]
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
        rows.append({
            "osm_id": way_id,
            "osm_type": "way",
            "geometry": geom_line,
            "tags": w.get("tags", {})
        })

    return rows


def _convert_relations_to_rows(
    relations: Dict[int, Dict[str, Any]],
    ways: Dict[int, Dict[str, Any]],
    nodes: Dict[int, Tuple[float, float]],
    simplify_multiline: bool = True
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

        rows.append({
            "osm_id": rel_id,
            "osm_type": "relation",
            "geometry": merged,
            "tags": r.get("tags", {})
        })

    return rows


def _build_osm_node_graph(
    gdf: gpd.GeoDataFrame,
    nodes: Dict[int, Tuple[float, float]],
    ways: Dict[int, Dict[str, Any]],
    geom: Union[Polygon, MultiPolygon]
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
    way_ids_in_gdf = set(gdf[gdf["osm_type"] == "way"]["osm_id"].values)

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


def get_river_network_from_shape(
    shp: Union[dict, Polygon, MultiPolygon],
    include_tags: Optional[Tuple[str, ...]] = None,
    overpass_endpoints: Optional[Iterable[str]] = None,
    timeout: int = 180,
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
    - include_tags: tuple of tag keys to match. Currently not used (waterway features are hardcoded).
                    Reserved for future extension.
    - overpass_endpoints: list/iterable of Overpass API endpoints to try. If None, a default list is used.
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
        raise ValueError("The provided shape must be a Polygon or MultiPolygon (or GeoJSON mapping).")

    if overpass_endpoints is None:
        overpass_endpoints = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.openstreetmap.ru/api/interpreter",
        ]

    waterway_values = "river|stream|canal|drain|riverbank|ditch|brook"

    # Choose query method: bbox is more robust, poly is more precise
    if use_bbox:
        # Use bounding box query
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        query = _build_overpass_bbox_query(bounds, waterway_values=waterway_values)
        logger.info("Querying Overpass for waterways in bbox: %s", bounds)
    else:
        # Use polygon query (more precise but can fail with large/complex polygons)
        try:
            poly_str = _make_overpass_poly(geom)
        except ValueError as e:
            raise ValueError(f"Failed to convert shape to Overpass poly format: {e}")

        try:
            query = _build_overpass_query(poly_str, waterway_values=waterway_values)
        except ValueError as e:
            raise ValueError(f"Failed to build Overpass query: {e}")

        logger.info("Querying Overpass for waterways in boundary (poly string: %d chars)", len(poly_str))

    # Fetch data
    data = _fetch_overpass(query, overpass_endpoints, timeout=timeout, max_retries=max_retries, backoff_factor=backoff_factor)
    elements = data.get("elements", [])

    # Aggregate elements into nodes, ways, and relations
    nodes, ways, relations = _aggregate_osm_elements(elements)

    # Convert OSM features to GeoDataFrame rows
    rows: List[Dict[str, Any]] = []
    rows.extend(_convert_ways_to_rows(ways, nodes))
    rows.extend(_convert_relations_to_rows(relations, ways, nodes, simplify_multiline=simplify_multiline))

    # Build GeoDataFrame
    if not rows:
        gdf = gpd.GeoDataFrame(columns=["osm_id", "osm_type", "geometry", "tags"], geometry="geometry", crs="EPSG:4326")
        if return_graph:
            return gdf, nx.Graph()
        return gdf

    gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs="EPSG:4326")

    # Clip to the provided polygon to ensure features are bounded by input
    try:
        gdf["geometry"] = gdf["geometry"].intersection(geom)
        gdf = gdf[~gdf["geometry"].is_empty]
    except Exception:
        logger.exception("Failed to intersect geometries with provided shape; returning un-clipped results.")

    gdf = gdf.reset_index(drop=True)

    if not return_graph:
        return gdf

    # Build OSM node-level graph
    G = _build_osm_node_graph(gdf, nodes, ways, geom)
    return gdf, G


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
            if 'length' in data:
                # Get node coordinates
                x1, y1 = G.nodes[u].get('x', 0), G.nodes[u].get('y', 0)
                x2, y2 = G.nodes[v].get('x', 0), G.nodes[v].get('y', 0)

                # Haversine formula
                lat1, lon1 = radians(y1), radians(x1)
                lat2, lon2 = radians(y2), radians(x2)

                dlat = lat2 - lat1
                dlon = lon2 - lon1

                a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                length_m = R * c

                total_length += length_m
    else:
        # Assume lengths are already in meters or similar linear units
        total_length = sum(data.get('length', 0) for u, v, data in G.edges(data=True))

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
    from shapely.geometry import LineString

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
    sindex = line_gdf.sindex

    for idx, row in line_gdf.iterrows():
        geom = row.geometry
        if not geom or geom.is_empty:
            continue

        # Find candidate intersecting lines using spatial index
        possible_matches_idx = list(sindex.intersection(geom.bounds))
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
    nodes_data = {}   # Maps node ID to coordinate
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

    # Process each line and split at nodes
    for orig_idx, row in line_gdf.iterrows():
        geom = row.geometry
        if not geom or geom.is_empty:
            continue

        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        # Find all nodes along this line
        nodes_on_line = []

        # Always include endpoints
        start_node = find_node_id(coords[0])
        end_node = find_node_id(coords[-1])
        nodes_on_line.append((0, start_node, coords[0]))

        # Check each coordinate to see if it's a node
        for i, coord in enumerate(coords[1:-1], start=1):
            if coord in node_id_map:
                node_id = node_id_map[coord]
                nodes_on_line.append((i, node_id, coord))

        nodes_on_line.append((len(coords) - 1, end_node, coords[-1]))

        # Create edges between consecutive nodes on this line
        for i in range(len(nodes_on_line) - 1):
            idx1, node1, coord1 = nodes_on_line[i]
            idx2, node2, coord2 = nodes_on_line[i + 1]

            if node1 == node2:
                continue

            # Extract segment coordinates
            segment_coords = coords[idx1:idx2 + 1]
            segment_geom = LineString(segment_coords)
            segment_length = segment_geom.length

            # Add edge (or update if already exists with shorter path)
            if G.has_edge(node1, node2):
                # Keep the shorter segment
                if segment_length < G[node1][node2].get(length_col, float('inf')):
                    G[node1][node2][length_col] = segment_length
                    G[node1][node2]['geometry'] = segment_geom
                    G[node1][node2]['source_idx'] = orig_idx
            else:
                G.add_edge(
                    node1,
                    node2,
                    **{
                        length_col: segment_length,
                        'geometry': segment_geom,
                        'source_idx': orig_idx
                    }
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
        path = nx.shortest_path(G, source_node, target_node, weight='length')
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
            x1, y1 = G.nodes[u].get('x', 0), G.nodes[u].get('y', 0)
            x2, y2 = G.nodes[v].get('x', 0), G.nodes[v].get('y', 0)

            # Haversine formula
            lat1, lon1 = radians(y1), radians(x1)
            lat2, lon2 = radians(y2), radians(x2)

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            length_m = R * c

            total_length += length_m
    else:
        # Assume lengths are already in meters or similar linear units
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            if G.has_edge(u, v):
                total_length += G[u][v].get('length', 0)

    return total_length

# Example usage
if __name__ == "__main__":
    import sys

    from shapely.geometry import box

    # Configure logging to see warnings/errors
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Example: small bounding box (adjust coordinates as needed)
    # This example uses a small area - bbox query is more reliable for testing
    bbox_poly = box(-0.12, 51.50, -0.10, 51.52)

    print("Fetching river network from Overpass API...")
    print("This may take 10-30 seconds depending on the area size and server load.")
    print("Using bbox query method for better reliability.\n")

    try:
        # Get GeoDataFrame only (not the OSM node-level graph)
        gdf = get_river_network_from_shape(
            bbox_poly,
            return_graph=False,
            max_retries=8,
            use_bbox=True
        )

        print(f"\nSuccess! Found {len(gdf)} river/stream features")
        print(f"  - Ways: {len(gdf[gdf['osm_type'] == 'way'])}")
        print(f"  - Relations: {len(gdf[gdf['osm_type'] == 'relation'])}")

        if len(gdf) > 0:
            print("\nFirst few features:")
            print(gdf[["osm_id", "osm_type", "tags"]].head())

            # Build topological network graph
            print("\n" + "="*60)
            print("Building topological network graph...")
            print("="*60)

            topo_graph = build_river_network_graph(gdf)

            print("\nTopological graph statistics:")
            print(f"  - Nodes (junctions/endpoints): {topo_graph.number_of_nodes()}")
            print(f"  - Edges (river segments): {topo_graph.number_of_edges()}")

            # Calculate total length
            if topo_graph.number_of_edges() > 0:
                total_length_deg = sum(d['length'] for u, v, d in topo_graph.edges(data=True))
                print(f"  - Total network length: {total_length_deg:.6f} degrees")

                # Calculate length in meters using Haversine
                total_length_m = calculate_graph_length_meters(topo_graph, crs=gdf.crs)
                print(f"  - Total network length: {total_length_m:.2f} meters ({total_length_m/1000:.2f} km)")

                # Analyze node degrees
                degrees = dict(topo_graph.degree())
                endpoints = sum(1 for d in degrees.values() if d == 1)
                junctions = sum(1 for d in degrees.values() if d == 2)
                confluences = sum(1 for d in degrees.values() if d >= 3)

                print("\nNode analysis:")
                print(f"  - Endpoints (degree 1): {endpoints}")
                print(f"  - Junctions (degree 2): {junctions}")
                print(f"  - Confluences (degree 3+): {confluences}")

                # Find connected components
                import networkx as nx
                components = list(nx.connected_components(topo_graph))
                print(f"  - Connected components: {len(components)}")
                if len(components) > 1:
                    component_sizes = sorted([len(c) for c in components], reverse=True)
                    print(f"    Largest component: {component_sizes[0]} nodes")

    except OverpassError as e:
        print(f"\nFailed to fetch data: {e}", file=sys.stderr)
        print("\nTroubleshooting tips:", file=sys.stderr)
        print("- Overpass servers may be overloaded; try again in a few minutes", file=sys.stderr)
        print("- Use a smaller bounding box", file=sys.stderr)
        print("- Try use_bbox=True for more reliable queries", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nInvalid input: {e}", file=sys.stderr)
        sys.exit(1)
