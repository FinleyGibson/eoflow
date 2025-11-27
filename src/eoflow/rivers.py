"""
River network utilities for fetching OpenStreetMap waterway data via Overpass API.

This module provides `get_river_network_from_shape` which queries the Overpass API
to retrieve river, stream, canal, and other waterway features within a boundary shape.

Key features:
- Accepts Shapely Polygon/MultiPolygon or GeoJSON-like mapping as boundary
- Queries Overpass with automatic retries and fallback endpoints (handles 504/5xx errors)
- Returns GeoPandas GeoDataFrame of LineString/MultiLineString geometries (EPSG:4326)
- Optionally returns NetworkX graph with node-level topology

Usage Examples
--------------

Basic usage with a bounding box::

    from shapely.geometry import box
    from eoflow.rivers import get_river_network_from_shape

    # Define area of interest
    bbox = box(-0.12, 51.50, -0.10, 51.52)  # London area

    # Fetch river network (returns GeoDataFrame)
    rivers_gdf = get_river_network_from_shape(bbox, use_bbox=True)
    print(f"Found {len(rivers_gdf)} waterway features")

    # Access feature attributes
    for idx, row in rivers_gdf.iterrows():
        print(f"OSM ID: {row['osm_id']}, Type: {row['osm_type']}")
        print(f"Tags: {row['tags']}")

With NetworkX graph for topology analysis::

    # Get both GeoDataFrame and graph
    gdf, graph = get_river_network_from_shape(
        bbox,
        return_graph=True,
        use_bbox=True
    )

    # Graph contains OSM nodes as vertices
    print(f"Network has {graph.number_of_nodes()} nodes")
    print(f"Network has {graph.number_of_edges()} edges")

    # Access node coordinates
    for node_id in list(graph.nodes())[:5]:
        x, y = graph.nodes[node_id]['x'], graph.nodes[node_id]['y']
        print(f"Node {node_id}: ({x}, {y})")

Using a custom polygon::

    from shapely.geometry import Polygon

    # Define custom polygon boundary
    coords = [(-0.12, 51.50), (-0.10, 51.50), (-0.10, 51.52), (-0.12, 51.52)]
    poly = Polygon(coords)

    # Query with polygon (more precise than bbox)
    gdf = get_river_network_from_shape(poly, use_bbox=False)

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
- Graph is built only from OSM ways (not relations) to preserve node-level topology
- Edge lengths in graph are in degrees; reproject to appropriate CRS for meters
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

    # Aggregate nodes/ways/relations
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

    rows: List[Dict[str, Any]] = []

    # Convert ways to LineStrings
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
        rows.append({"osm_id": way_id, "osm_type": "way", "geometry": geom_line, "tags": w.get("tags", {})})

    # Convert relations: stitch member ways
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
            # Keep all lines as a MultiLineString if there are multiple, otherwise single LineString
            merged = MultiLineString(lines) if len(lines) > 1 else lines[0]
        rows.append({"osm_id": rel_id, "osm_type": "relation", "geometry": merged, "tags": r.get("tags", {})})

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

    # Build NetworkX graph: nodes are OSM node ids, edges are consecutive node pairs from ways
    # Only include ways that are actually in the final GeoDataFrame
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
                # length stored in degrees; reproject if you need meters
                seg_length = edge_geom.length
                G.add_edge(u, v, way_id=way_id, tags=w.get("tags", {}), length=seg_length)

    return gdf, G


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
        # Get GeoDataFrame and NetworkX graph
        # use_bbox=True is more reliable for simple rectangular areas
        gdf, G = get_river_network_from_shape(
            bbox_poly,
            return_graph=True,
            max_retries=8,  # Increase retries for robustness
            use_bbox=True   # Use bbox query (more robust than poly)
        )

        print(f"\nSuccess! Found {len(gdf)} river/stream features")
        print(f"  - Ways: {len(gdf[gdf['osm_type'] == 'way'])}")
        print(f"  - Relations: {len(gdf[gdf['osm_type'] == 'relation'])}")
        print("\nGraph statistics:")
        print(f"  - Nodes: {G.number_of_nodes()}")
        print(f"  - Edges: {G.number_of_edges()}")

        if len(gdf) > 0:
            print("\nFirst few features:")
            print(gdf[["osm_id", "osm_type", "tags"]].head())

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
