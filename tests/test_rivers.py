"""
Tests for eoflow.rivers.

The tests are split into two groups:

- Unit tests (the majority of this file) exercise pure/parsing logic and use
  mocked HTTP responses, so they run quickly and require no network access.
- Integration tests hit the real Overpass API and are marked with
  `@pytest.mark.integration` (see tests/INTEGRATION_TESTS.md). Run them with
  `pytest -m integration`; they are skipped by default via
  `pytest -m "not integration"`.
"""

import math
from unittest.mock import Mock, patch

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
import requests
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box, mapping
from shapely.geometry import shape as shapely_shape

from eoflow.rivers import (
    _REQUEST_HEADERS,
    OVERPASS_ENDPOINTS,
    OverpassError,
    _aggregate_osm_elements,
    _build_osm_node_graph,
    _build_overpass_bbox_query,
    _build_overpass_query_from_poly_str,
    _convert_relations_to_rows,
    _convert_ways_to_rows,
    _fetch_overpass,
    _make_overpass_poly,
    build_river_network_graph,
    calculate_graph_length_meters,
    calculate_shortest_path_length,
    get_river_network_from_poly,
)

EXETER_POLY = Polygon(
    (
        (-3.5339, 50.7284),
        (-3.5252, 50.7234),
        (-3.5252, 50.7134),
        (-3.5339, 50.7084),
        (-3.5425, 50.7134),
        (-3.5425, 50.7234),
        (-3.5339, 50.7284),
    )
)


def wkt_to_overpass(s: str) -> str:
    """
    Convert a WKT polygon string to an Overpass query string.
    """
    return " ".join(
        f"{lat} {lon}"
        for lon, lat in (
            p.split() for p in s.removeprefix("POLYGON ((").removesuffix("))").split(", ")
        )
    )


def _mock_response(status_code=200, json_data=None, text="", raise_json_error=False):
    """Build a Mock object that looks like a `requests.Response`."""
    resp = Mock()
    resp.status_code = status_code
    resp.text = text
    if raise_json_error:
        resp.json.side_effect = ValueError("No JSON object could be decoded")
    else:
        resp.json.return_value = json_data if json_data is not None else {"elements": []}
    resp.raise_for_status = Mock()
    return resp


# ---------------------------------------------------------------------------
# Integration tests: real Overpass API calls
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRivers:
    @staticmethod
    def wkt_to_overpass(s: str) -> str:
        return " ".join(
            f"{lat} {lon}"
            for lon, lat in (
                p.split() for p in s.removeprefix("POLYGON ((").removesuffix("))").split(", ")
            )
        )

    @classmethod
    def setup_class(cls):
        # Example shape in lat/long50.7184, -3.5339
        cls.shp = EXETER_POLY
        cls.geom = shapely_shape(cls.shp)
        cls.waterway_values = "river|stream|canal|drain|riverbank|ditch|brook"
        cls.bounds = cls.geom.bounds  # (min_x, min_y, max_x, max_y)
        cls.overpass_endpoints = OVERPASS_ENDPOINTS

    @pytest.mark.parametrize("endpoint", OVERPASS_ENDPOINTS)
    def test_bare_bbox_request(self, endpoint):
        """
        This test passes when the Overpass API returns a valid response for a poly query, but this is inconsistent.
        """
        query = _build_overpass_bbox_query(self.bounds, waterway_values=self.waterway_values)
        resp = requests.post(endpoint, data={"data": query}, timeout=180, headers=_REQUEST_HEADERS)

        assert resp.status_code < 400

    @pytest.mark.parametrize("endpoint", OVERPASS_ENDPOINTS)
    def test_bare_poly_request(self, endpoint):
        """
        This test passes when the Overpass API returns a valid response for a poly query, but this is inconsistent.
        """
        poly_str_ = self.shp.wkt
        poly_str = self.wkt_to_overpass(poly_str_)

        query = _build_overpass_query_from_poly_str(
            poly_str=poly_str, waterway_values=self.waterway_values
        )
        resp = requests.post(endpoint, data={"data": query}, timeout=180, headers=_REQUEST_HEADERS)

        assert resp.status_code < 400

    def test_bbox_query_building(self):
        query = _build_overpass_bbox_query(self.bounds, waterway_values=self.waterway_values)

        assert isinstance(query, str)
        assert "way[" in query

        assert "relation[" in query
        assert query.count("(") == query.count(")")

    def test_poly_query_building(self):
        query = _build_overpass_query_from_poly_str(
            poly_str=self.wkt_to_overpass(self.shp.wkt), waterway_values=self.waterway_values
        )

        assert isinstance(query, str)
        assert "way[" in query
        assert "poly:" in query

    def test_fetch_overpass_bbox(self):
        query = _build_overpass_bbox_query(self.bounds, waterway_values=self.waterway_values)

        # Fetch data
        data = _fetch_overpass(
            query,
            self.overpass_endpoints,
            timeout=180,
            max_retries=3,
            backoff_factor=1.5,
        )

        assert data["elements"]
        assert isinstance(data["elements"], list)
        assert len(data["elements"]) > 0

    def test_fetch_overpass_poly(self):
        query = _build_overpass_query_from_poly_str(
            poly_str=self.wkt_to_overpass(self.shp.wkt),
            waterway_values=self.waterway_values,
        )

        # Fetch data
        data = _fetch_overpass(
            query,
            self.overpass_endpoints,
            timeout=60,
            max_retries=3,
            backoff_factor=1.5,
        )

        assert isinstance(data, dict)
        assert "elements" in data

    def test_gdf_from_poly_bbox(self):
        rivers, graph = get_river_network_from_poly(self.shp, return_graph=True, use_bbox=True)

        # Check that the GDF is returned and has the expected columns
        assert isinstance(rivers, gpd.GeoDataFrame)
        assert rivers.shape[0] > 0
        for col in ["osm_id", "osm_type", "geometry", "tags"]:
            assert col in rivers.columns

        # Check that the graph is returned and has edges
        assert isinstance(graph, nx.Graph)
        assert graph.edges

    def test_gdf_from_poly(self):
        rivers, graph = get_river_network_from_poly(self.shp, return_graph=True, use_bbox=False)

        # Check that the GDF is returned and has the expected columns
        assert isinstance(rivers, gpd.GeoDataFrame)
        assert rivers.shape[0] > 0
        for col in ["osm_id", "osm_type", "geometry", "tags"]:
            assert col in rivers.columns

        # Check that the graph is returned and has edges
        assert isinstance(graph, nx.Graph)
        assert graph.edges


# ---------------------------------------------------------------------------
# Unit tests: query builders
# ---------------------------------------------------------------------------


class TestMakeOverpassPoly:
    def test_single_polygon(self):
        p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        poly_str = _make_overpass_poly(p)

        # First coordinate pair should be "lat lon" (y then x)
        assert poly_str.startswith("0.000000 0.000000")
        assert ";" not in poly_str

    def test_multipolygon_joins_with_semicolon(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        p2 = Polygon([(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)])
        mp = MultiPolygon([p1, p2])

        poly_str = _make_overpass_poly(mp)

        assert poly_str.count(" ; ") == 1
        parts = poly_str.split(" ; ")
        assert len(parts) == 2

    def test_raises_value_error_when_no_valid_polygons(self):
        # A polygon can't legally have < 3 coords, but guard against
        # degenerate/mocked input reaching the exterior-coords check.
        p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        with patch.object(
            type(p.exterior), "coords", new_callable=lambda: property(lambda self: [(0, 0), (1, 1)])
        ):
            with pytest.raises(ValueError):
                _make_overpass_poly(p)


class TestBuildOverpassQueryFromPolyStr:
    def test_builds_valid_query(self):
        query = _build_overpass_query_from_poly_str("0 0 1 0 1 1 0 0", waterway_values="river")

        assert "poly:" in query
        assert 'waterway"~"river"' in query
        assert query.count("(") == query.count(")")

    def test_empty_poly_str_raises_value_error(self):
        with pytest.raises(ValueError):
            _build_overpass_query_from_poly_str("", waterway_values="river")

    def test_whitespace_only_poly_str_raises_value_error(self):
        with pytest.raises(ValueError):
            _build_overpass_query_from_poly_str("   ", waterway_values="river")


class TestBuildOverpassBboxQuery:
    def test_builds_valid_query_with_correct_bounds(self):
        bbox = (-1.0, 50.0, 1.0, 51.0)  # (min_lon, min_lat, max_lon, max_lat)
        query = _build_overpass_bbox_query(bbox, waterway_values="river|stream")

        assert isinstance(query, str)
        assert "way[" in query
        assert "relation[" in query
        # bbox in Overpass order is (min_lat, min_lon, max_lat, max_lon)
        assert "(50.0,-1.0,51.0,1.0)" in query
        assert query.count("(") == query.count(")")


# ---------------------------------------------------------------------------
# Unit tests: _fetch_overpass retry/error handling (mocked HTTP)
# ---------------------------------------------------------------------------


class TestFetchOverpass:
    def test_no_endpoints_raises_value_error(self):
        with pytest.raises(ValueError):
            _fetch_overpass("query", [], max_retries=1)

    @patch("eoflow.rivers.time.sleep", return_value=None)
    @patch("eoflow.rivers.requests.post")
    def test_success_on_first_attempt(self, mock_post, mock_sleep):
        mock_post.return_value = _mock_response(json_data={"elements": [{"type": "node"}]})

        data = _fetch_overpass("query", ["http://endpoint-a"], max_retries=3)

        assert data == {"elements": [{"type": "node"}]}
        mock_post.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("eoflow.rivers.time.sleep", return_value=None)
    @patch("eoflow.rivers.requests.post")
    def test_succeeds_after_transient_failure(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            _mock_response(status_code=502),
            _mock_response(json_data={"elements": []}),
        ]

        data = _fetch_overpass("query", ["http://endpoint-a"], max_retries=3)

        assert data == {"elements": []}
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()

    @patch("eoflow.rivers.time.sleep", return_value=None)
    @patch("eoflow.rivers.requests.post")
    def test_exhausts_retries_on_persistent_server_errors(self, mock_post, mock_sleep):
        mock_post.return_value = _mock_response(status_code=504)

        with pytest.raises(OverpassError):
            _fetch_overpass("query", ["http://endpoint-a"], max_retries=3)

        assert mock_post.call_count == 3

    @patch("eoflow.rivers.time.sleep", return_value=None)
    @patch("eoflow.rivers.requests.post")
    def test_bad_query_400_eventually_raises_overpass_error(self, mock_post, mock_sleep):
        mock_post.return_value = _mock_response(status_code=400, text="bad query")

        with pytest.raises(OverpassError):
            _fetch_overpass("query", ["http://endpoint-a"], max_retries=2)

        assert mock_post.call_count == 2

    @patch("eoflow.rivers.time.sleep", return_value=None)
    @patch("eoflow.rivers.requests.post")
    def test_invalid_json_raises_overpass_error(self, mock_post, mock_sleep):
        mock_post.return_value = _mock_response(
            status_code=200, text="<html>error</html>", raise_json_error=True
        )

        with pytest.raises(OverpassError):
            _fetch_overpass("query", ["http://endpoint-a"], max_retries=2)

        assert mock_post.call_count == 2

    @patch("eoflow.rivers.time.sleep", return_value=None)
    @patch("eoflow.rivers.requests.post")
    def test_connection_error_retries_then_raises(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.exceptions.ConnectionError("boom")

        with pytest.raises(OverpassError):
            _fetch_overpass("query", ["http://endpoint-a"], max_retries=2)

        assert mock_post.call_count == 2

    @patch("eoflow.rivers.time.sleep", return_value=None)
    @patch("eoflow.rivers.requests.post")
    def test_timeout_retries_then_raises(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")

        with pytest.raises(OverpassError):
            _fetch_overpass("query", ["http://endpoint-a"], max_retries=2)

        assert mock_post.call_count == 2

    @patch("eoflow.rivers.time.sleep", return_value=None)
    @patch("eoflow.rivers.requests.post")
    def test_rotates_across_multiple_endpoints(self, mock_post, mock_sleep):
        mock_post.return_value = _mock_response(status_code=503)
        endpoints = ["http://endpoint-a", "http://endpoint-b"]

        with pytest.raises(OverpassError):
            _fetch_overpass("query", endpoints, max_retries=4)

        called_endpoints = [call.args[0] for call in mock_post.call_args_list]
        assert called_endpoints == [
            "http://endpoint-a",
            "http://endpoint-b",
            "http://endpoint-a",
            "http://endpoint-b",
        ]


# ---------------------------------------------------------------------------
# Unit tests: OSM element parsing helpers
# ---------------------------------------------------------------------------


class TestAggregateOsmElements:
    def test_separates_nodes_ways_and_relations(self):
        elements = [
            {"type": "node", "id": 1, "lon": 0.0, "lat": 0.0},
            {"type": "node", "id": 2, "lon": 1.0, "lat": 1.0},
            {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"waterway": "river"}},
            {
                "type": "relation",
                "id": 100,
                "members": [{"type": "way", "ref": 10}],
                "tags": {"waterway": "river"},
            },
            {"type": "unknown", "id": 999},
        ]

        nodes, ways, relations = _aggregate_osm_elements(elements)

        assert nodes == {1: (0.0, 0.0), 2: (1.0, 1.0)}
        assert set(ways.keys()) == {10}
        assert set(relations.keys()) == {100}

    def test_node_without_coords_is_skipped(self):
        elements = [{"type": "node", "id": 1}]

        nodes, ways, relations = _aggregate_osm_elements(elements)

        assert nodes == {}

    def test_empty_elements_list(self):
        nodes, ways, relations = _aggregate_osm_elements([])

        assert nodes == {} and ways == {} and relations == {}


class TestConvertWaysToRows:
    def test_valid_way_produces_linestring_row(self):
        nodes = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0)}
        ways = {10: {"nodes": [1, 2, 3], "tags": {"waterway": "river"}}}

        rows = _convert_ways_to_rows(ways, nodes)

        assert len(rows) == 1
        row = rows[0]
        assert row["osm_id"] == 10
        assert row["osm_type"] == "way"
        assert row["tags"] == {"waterway": "river"}
        assert isinstance(row["geometry"], LineString)
        assert list(row["geometry"].coords) == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]

    def test_way_with_missing_node_is_skipped(self):
        nodes = {1: (0.0, 0.0)}
        ways = {11: {"nodes": [1, 99], "tags": {}}}

        rows = _convert_ways_to_rows(ways, nodes)

        assert rows == []

    def test_way_with_fewer_than_two_valid_nodes_is_skipped(self):
        nodes = {1: (0.0, 0.0)}
        ways = {12: {"nodes": [1], "tags": {}}}

        rows = _convert_ways_to_rows(ways, nodes)

        assert rows == []

    def test_multiple_ways_only_valid_ones_returned(self):
        nodes = {1: (0.0, 0.0), 2: (1.0, 0.0)}
        ways = {
            10: {"nodes": [1, 2], "tags": {}},
            11: {"nodes": [1, 99], "tags": {}},
        }

        rows = _convert_ways_to_rows(ways, nodes)

        assert len(rows) == 1
        assert rows[0]["osm_id"] == 10


class TestConvertRelationsToRows:
    def test_contiguous_ways_merged_into_single_linestring(self):
        nodes = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0)}
        ways = {20: {"nodes": [1, 2], "tags": {}}, 21: {"nodes": [2, 3], "tags": {}}}
        relations = {
            100: {
                "members": [{"type": "way", "ref": 20}, {"type": "way", "ref": 21}],
                "tags": {"waterway": "river"},
            }
        }

        rows = _convert_relations_to_rows(relations, ways, nodes, simplify_multiline=True)

        assert len(rows) == 1
        row = rows[0]
        assert row["osm_id"] == 100
        assert row["osm_type"] == "relation"
        assert row["geometry"].geom_type == "LineString"

    def test_relation_with_missing_way_reference_is_skipped(self):
        nodes = {1: (0.0, 0.0), 2: (1.0, 0.0)}
        ways = {20: {"nodes": [1, 2], "tags": {}}}
        relations = {101: {"members": [{"type": "way", "ref": 999}], "tags": {}}}

        rows = _convert_relations_to_rows(relations, ways, nodes)

        assert rows == []

    def test_relation_where_member_way_has_no_valid_coords_is_skipped(self):
        nodes = {1: (0.0, 0.0)}
        ways = {22: {"nodes": [10, 11], "tags": {}}}  # neither node resolvable
        relations = {102: {"members": [{"type": "way", "ref": 22}], "tags": {}}}

        rows = _convert_relations_to_rows(relations, ways, nodes)

        assert rows == []

    def test_disjoint_ways_without_simplify_produce_multilinestring(self):
        nodes = {1: (0.0, 0.0), 2: (1.0, 0.0), 5: (10.0, 10.0), 6: (11.0, 10.0)}
        ways = {23: {"nodes": [1, 2], "tags": {}}, 24: {"nodes": [5, 6], "tags": {}}}
        relations = {
            103: {
                "members": [{"type": "way", "ref": 23}, {"type": "way", "ref": 24}],
                "tags": {},
            }
        }

        rows = _convert_relations_to_rows(relations, ways, nodes, simplify_multiline=False)

        assert len(rows) == 1
        assert isinstance(rows[0]["geometry"], MultiLineString)

    def test_relation_with_no_way_members_is_skipped(self):
        nodes = {1: (0.0, 0.0)}
        ways = {}
        relations = {104: {"members": [{"type": "node", "ref": 1}], "tags": {}}}

        rows = _convert_relations_to_rows(relations, ways, nodes)

        assert rows == []


# ---------------------------------------------------------------------------
# Unit tests: OSM node-level graph building
# ---------------------------------------------------------------------------


class TestBuildOsmNodeGraph:
    def test_builds_edges_only_for_ways_present_in_gdf(self):
        nodes = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0), 4: (2.0, 1.0)}
        ways = {
            10: {"nodes": [1, 2, 3], "tags": {"waterway": "river"}},
            11: {"nodes": [3, 4], "tags": {"waterway": "stream"}},
        }
        gdf = pd.DataFrame({"osm_id": [10], "osm_type": ["way"]})
        geom = box(-1, -1, 10, 10)

        G = _build_osm_node_graph(gdf, nodes, ways, geom)

        assert set(G.nodes) == {1, 2, 3}
        assert set(G.edges) == {(1, 2), (2, 3)}
        assert G[1][2]["way_id"] == 10
        assert G[1][2]["tags"] == {"waterway": "river"}
        assert G[1][2]["length"] == pytest.approx(1.0)

    def test_edges_outside_boundary_are_excluded(self):
        nodes = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (50.0, 50.0), 4: (60.0, 60.0)}
        ways = {
            10: {"nodes": [1, 2], "tags": {}},
            11: {"nodes": [3, 4], "tags": {}},
        }
        gdf = pd.DataFrame({"osm_id": [10, 11], "osm_type": ["way", "way"]})
        geom = box(-1, -1, 2, 2)

        G = _build_osm_node_graph(gdf, nodes, ways, geom)

        assert set(G.edges) == {(1, 2)}

    def test_node_coordinates_stored_as_attributes(self):
        nodes = {1: (0.0, 0.0), 2: (1.0, 0.0)}
        ways = {10: {"nodes": [1, 2], "tags": {}}}
        gdf = pd.DataFrame({"osm_id": [10], "osm_type": ["way"]})
        geom = box(-1, -1, 2, 2)

        G = _build_osm_node_graph(gdf, nodes, ways, geom)

        assert G.nodes[1]["x"] == 0.0
        assert G.nodes[1]["y"] == 0.0

    def test_empty_ways_produces_empty_graph(self):
        gdf = pd.DataFrame({"osm_id": [], "osm_type": []})
        geom = box(-1, -1, 2, 2)

        G = _build_osm_node_graph(gdf, {}, {}, geom)

        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


# ---------------------------------------------------------------------------
# Unit tests: get_river_network_from_poly (mocked Overpass calls)
# ---------------------------------------------------------------------------


class TestGetRiverNetworkFromPoly:
    def test_invalid_shape_type_raises_value_error(self):
        with pytest.raises(ValueError):
            get_river_network_from_poly("not-a-shape")

    @patch("eoflow.rivers._fetch_overpass")
    def test_accepts_geojson_mapping_input(self, mock_fetch):
        mock_fetch.return_value = {"elements": []}

        gdf = get_river_network_from_poly(mapping(EXETER_POLY), use_bbox=True)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.shape[0] == 0
        for col in ["osm_id", "osm_type", "geometry", "tags"]:
            assert col in gdf.columns

    @patch("eoflow.rivers._fetch_overpass")
    def test_empty_overpass_response_returns_empty_gdf_and_graph(self, mock_fetch):
        mock_fetch.return_value = {"elements": []}

        gdf, graph = get_river_network_from_poly(EXETER_POLY, return_graph=True, use_bbox=True)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.shape[0] == 0
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 0

    @patch("eoflow.rivers._build_overpass_bbox_query")
    @patch("eoflow.rivers._fetch_overpass")
    def test_use_bbox_true_builds_bbox_query(self, mock_fetch, mock_bbox_query):
        mock_fetch.return_value = {"elements": []}
        mock_bbox_query.return_value = "QUERY"

        get_river_network_from_poly(EXETER_POLY, use_bbox=True)

        mock_bbox_query.assert_called_once()

    @patch("eoflow.rivers._build_overpass_query_from_poly_str")
    @patch("eoflow.rivers._fetch_overpass")
    def test_use_bbox_false_builds_poly_query(self, mock_fetch, mock_poly_query):
        mock_fetch.return_value = {"elements": []}
        mock_poly_query.return_value = "QUERY"

        get_river_network_from_poly(EXETER_POLY, use_bbox=False)

        mock_poly_query.assert_called_once()

    @patch("eoflow.rivers._fetch_overpass")
    def test_builds_gdf_and_graph_from_synthetic_response(self, mock_fetch):
        # A single way running through the middle of EXETER_POLY's bbox.
        min_x, min_y, max_x, max_y = EXETER_POLY.bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        mock_fetch.return_value = {
            "elements": [
                {"type": "node", "id": 1, "lon": mid_x - 0.001, "lat": mid_y},
                {"type": "node", "id": 2, "lon": mid_x, "lat": mid_y},
                {"type": "node", "id": 3, "lon": mid_x + 0.001, "lat": mid_y},
                {
                    "type": "way",
                    "id": 500,
                    "nodes": [1, 2, 3],
                    "tags": {"waterway": "river", "name": "Test River"},
                },
            ]
        }

        gdf, graph = get_river_network_from_poly(EXETER_POLY, return_graph=True, use_bbox=True)

        assert gdf.shape[0] == 1
        assert gdf.iloc[0]["osm_id"] == 500
        assert gdf.iloc[0]["osm_type"] == "way"
        assert gdf.iloc[0]["tags"] == {"waterway": "river", "name": "Test River"}

        assert graph.number_of_edges() == 2

    @patch("eoflow.rivers._fetch_overpass")
    def test_without_return_graph_returns_only_gdf(self, mock_fetch):
        mock_fetch.return_value = {"elements": []}

        result = get_river_network_from_poly(EXETER_POLY, return_graph=False, use_bbox=True)

        assert isinstance(result, gpd.GeoDataFrame)


# ---------------------------------------------------------------------------
# Unit tests: length calculations
# ---------------------------------------------------------------------------


def _haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, (lat1, lon1, lat2, lon2))
    d_lat = lat2_r - lat1_r
    d_lon = lon2_r - lon1_r
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(d_lon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class TestCalculateGraphLengthMeters:
    def test_geographic_crs_uses_haversine(self):
        G = nx.Graph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=0.0, y=1.0)
        G.add_edge(1, 2, length=1.0)  # degrees; ignored in favour of haversine

        total = calculate_graph_length_meters(G, crs="EPSG:4326")

        expected = _haversine_m(0.0, 0.0, 0.0, 1.0)
        assert total == pytest.approx(expected, rel=1e-6)

    def test_non_geographic_crs_sums_length_attribute_directly(self):
        G = nx.Graph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=100.0, y=0.0)
        G.add_node(3, x=100.0, y=50.0)
        G.add_edge(1, 2, length=100.0)
        G.add_edge(2, 3, length=50.0)

        total = calculate_graph_length_meters(G, crs="EPSG:27700")

        assert total == pytest.approx(150.0)

    def test_empty_graph_returns_zero(self):
        assert calculate_graph_length_meters(nx.Graph()) == 0.0


class TestCalculateShortestPathLength:
    def test_geographic_path_length(self):
        G = nx.Graph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=0.0, y=1.0)
        G.add_node(3, x=0.0, y=2.0)
        G.add_edge(1, 2, length=1.0)
        G.add_edge(2, 3, length=1.0)

        total = calculate_shortest_path_length(G, 1, 3, crs="EPSG:4326")

        expected = _haversine_m(0.0, 0.0, 0.0, 1.0) + _haversine_m(0.0, 1.0, 0.0, 2.0)
        assert total == pytest.approx(expected, rel=1e-6)

    def test_non_geographic_path_length(self):
        G = nx.Graph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=10.0, y=0.0)
        G.add_node(3, x=25.0, y=0.0)
        G.add_edge(1, 2, length=10.0)
        G.add_edge(2, 3, length=15.0)

        total = calculate_shortest_path_length(G, 1, 3, crs="EPSG:27700")

        assert total == pytest.approx(25.0)

    def test_prefers_shorter_of_multiple_paths(self):
        G = nx.Graph()
        for n in (1, 2, 3, 4):
            G.add_node(n, x=float(n), y=0.0)
        G.add_edge(1, 4, length=100.0)
        G.add_edge(1, 2, length=1.0)
        G.add_edge(2, 3, length=1.0)
        G.add_edge(3, 4, length=1.0)

        total = calculate_shortest_path_length(G, 1, 4, crs="EPSG:27700")

        assert total == pytest.approx(3.0)

    def test_missing_source_node_raises(self):
        G = nx.Graph()
        G.add_node(2, x=0.0, y=0.0)

        with pytest.raises(nx.NodeNotFound):
            calculate_shortest_path_length(G, 1, 2)

    def test_missing_target_node_raises(self):
        G = nx.Graph()
        G.add_node(1, x=0.0, y=0.0)

        with pytest.raises(nx.NodeNotFound):
            calculate_shortest_path_length(G, 1, 2)

    def test_no_path_raises(self):
        G = nx.Graph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=1.0, y=0.0)
        # No edge between 1 and 2 -> disconnected

        with pytest.raises(nx.NetworkXNoPath):
            calculate_shortest_path_length(G, 1, 2)


# ---------------------------------------------------------------------------
# Unit tests: topological graph building
# ---------------------------------------------------------------------------


class TestBuildRiverNetworkGraph:
    def test_no_linestrings_returns_empty_graph(self):
        gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

        G = build_river_network_graph(gdf)

        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 0

    def test_shared_endpoint_creates_single_junction_node(self):
        lines = [
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (1, 1)]),
        ]
        gdf = gpd.GeoDataFrame({"geometry": lines}, crs="EPSG:4326")

        G = build_river_network_graph(gdf)

        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        # The junction node at (1, 0) should have degree 2
        junction = [n for n, d in G.nodes(data=True) if (d["x"], d["y"]) == (1, 0)]
        assert len(junction) == 1
        assert G.degree(junction[0]) == 2

    def test_crossing_lines_without_shared_vertex_are_split(self):
        # build_river_network_graph detects the geometric intersection point
        # of these two crossing lines and splits both of them there, even
        # though neither line has an original vertex at (1, 1).
        lines = [
            LineString([(0, 0), (2, 2)]),
            LineString([(0, 2), (2, 0)]),
        ]
        gdf = gpd.GeoDataFrame({"geometry": lines}, crs="EPSG:4326")

        G = build_river_network_graph(gdf)

        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 4
        crossing = [n for n, d in G.nodes(data=True) if (d["x"], d["y"]) == (1.0, 1.0)]
        assert len(crossing) == 1
        assert G.degree(crossing[0]) == 4

    def test_crossing_lines_with_shared_vertex_are_split(self):
        # When the crossing point IS an actual vertex of both lines, it is
        # also correctly used to split them into separate edges.
        lines = [
            LineString([(0, 0), (1, 1), (2, 2)]),
            LineString([(0, 2), (1, 1), (2, 0)]),
        ]
        gdf = gpd.GeoDataFrame({"geometry": lines}, crs="EPSG:4326")

        G = build_river_network_graph(gdf)

        assert G.number_of_nodes() == 5
        assert G.number_of_edges() == 4
        crossing = [n for n, d in G.nodes(data=True) if (d["x"], d["y"]) == (1.0, 1.0)]
        assert len(crossing) == 1
        assert G.degree(crossing[0]) == 4

    def test_edge_length_matches_segment_length(self):
        lines = [LineString([(0, 0), (3, 4)])]  # length 5 (3-4-5 triangle)
        gdf = gpd.GeoDataFrame({"geometry": lines}, crs="EPSG:4326")

        G = build_river_network_graph(gdf)

        assert G.number_of_edges() == 1
        (u, v, data) = list(G.edges(data=True))[0]
        assert data["length"] == pytest.approx(5.0)
        assert isinstance(data["geometry"], LineString)

    def test_disjoint_lines_produce_separate_components(self):
        lines = [
            LineString([(0, 0), (1, 0)]),
            LineString([(10, 10), (11, 10)]),
        ]
        gdf = gpd.GeoDataFrame({"geometry": lines}, crs="EPSG:4326")

        G = build_river_network_graph(gdf)

        assert nx.number_connected_components(G) == 2

    def test_multilinestring_input_is_exploded(self):
        mls = MultiLineString([[(0, 0), (1, 0)], [(2, 0), (3, 0)]])
        gdf = gpd.GeoDataFrame({"geometry": [mls]}, crs="EPSG:4326")

        G = build_river_network_graph(gdf)

        assert G.number_of_edges() == 2
        assert nx.number_connected_components(G) == 2


if __name__ == "__main__":
    pytest.main()
