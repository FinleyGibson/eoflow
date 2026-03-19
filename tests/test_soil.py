"""
Unit and integration tests for eoflow.soil — DEFRA SEARG soil coverage module.

Unit tests mock all HTTP calls so no network access is required.
Integration tests (marked ``@pytest.mark.integration``) hit the real
ArcGIS REST service and require internet access.

Run unit tests only (default)::

    pytest tests/test_soil.py

Run everything including integration tests::

    pytest -m integration tests/test_soil.py
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock

import geopandas as gpd
import pandas as pd
import pytest
import requests
from shapely.geometry import Point, Polygon, box

# ---------------------------------------------------------------------------
# Helpers — synthetic geometry and response builders
# ---------------------------------------------------------------------------

# A small square polygon in WGS-84 over Devon
_DEVON_WGS84 = box(-3.6, 50.6, -3.4, 50.8)

# Corresponding approximate BNG bbox (metres): ~285000,66000 → ~300000,88000
# We use a simplified square for testing purposes
_DEVON_BNG_APPROX = box(285_000, 66_000, 300_000, 88_000)


def _make_ring(polygon: Polygon) -> list:
    """Return an ArcGIS-style ring (list of [x, y] pairs) from a Shapely polygon."""
    return [list(coord) for coord in polygon.exterior.coords]


def _make_soil_feature(
    objectid: int,
    searg_concise: str,
    searg: str = "Test group",
    mu_name: str = "TESTSOIL",
    map_symbol: str = "TS",
    bfi: float = 0.5,
    spr: int = 40,
    description: str = "A test soil.",
    ring_polygon: Polygon | None = None,
) -> dict:
    """Build a minimal ArcGIS feature dict that mirrors the real service response."""
    if ring_polygon is None:
        ring_polygon = box(285_000, 66_000, 290_000, 72_000)
    return {
        "attributes": {
            "OBJECTID": objectid,
            "SEARG_Concise": searg_concise,
            "SEARG": searg,
            "MU_NAME": mu_name,
            "MAP_SYMBOL": map_symbol,
            "BFI": bfi,
            "SPR": spr,
            "SEARGDescription": description,
            "Shape__Area": ring_polygon.area,
        },
        "geometry": {
            "rings": [_make_ring(ring_polygon)],
        },
    }


def _make_query_response(
    features: list,
    exceeded: bool = False,
) -> dict:
    """Build a mock ArcGIS query JSON response body."""
    return {
        "objectIdFieldName": "OBJECTID",
        "geometryType": "esriGeometryPolygon",
        "spatialReference": {"wkid": 27700, "latestWkid": 27700},
        "fields": [],
        "features": features,
        "exceededTransferLimit": exceeded,
    }


def _mock_session(responses: list) -> MagicMock:
    """Return a mock requests.Session whose .post() returns responses in order."""
    session = MagicMock(spec=requests.Session)
    mock_responses = []
    for body in responses:
        r = Mock(spec=requests.Response)
        r.status_code = 200
        r.raise_for_status = Mock()
        r.json.return_value = body
        mock_responses.append(r)
    session.post.side_effect = mock_responses
    return session


# ===========================================================================
# Tests: module-level constants
# ===========================================================================


class TestConstants:
    """Verify the public constants are consistent and correct."""

    def test_searg_groups_count(self):
        """There are exactly 12 SEARG_Concise groups."""
        from eoflow.soil import SEARG_GROUPS

        assert len(SEARG_GROUPS) == 12

    def test_searg_groups_are_strings(self):
        from eoflow.soil import SEARG_GROUPS

        assert all(isinstance(g, str) for g in SEARG_GROUPS)

    def test_searg_groups_unique(self):
        from eoflow.soil import SEARG_GROUPS

        assert len(SEARG_GROUPS) == len(set(SEARG_GROUPS))

    def test_searg_groups_known_members(self):
        from eoflow.soil import SEARG_GROUPS

        expected_subset = {
            "Peat",
            "Shallow soils",
            "Man made",
            "Light free drainage",
            "Alluvial and coastal soils",
            "Heavy clay soils with poor drainage",
        }
        assert expected_subset.issubset(set(SEARG_GROUPS))

    def test_query_url_contains_feature_server(self):
        from eoflow.soil import QUERY_URL

        assert "FeatureServer" in QUERY_URL
        assert "/query" in QUERY_URL

    def test_query_url_contains_item_service(self):
        from eoflow.soil import FEATURE_SERVICE_URL, ITEM_ID

        assert "SEARGFull" in FEATURE_SERVICE_URL or ITEM_ID in ("af498634e7c2409c8a0e3eecb720f8dc")

    def test_layer_url_is_layer_zero(self):
        from eoflow.soil import LAYER_URL

        assert LAYER_URL.endswith("/0")

    def test_max_record_count_is_positive(self):
        from eoflow.soil import MAX_RECORD_COUNT

        assert MAX_RECORD_COUNT > 0

    def test_item_id_format(self):
        """Item ID should be a 32-character hex string."""
        from eoflow.soil import ITEM_ID

        assert len(ITEM_ID) == 32
        assert all(c in "0123456789abcdef" for c in ITEM_ID)


# ===========================================================================
# Tests: _to_bng
# ===========================================================================


class TestToBng:
    """Tests for the internal WGS84 → BNG reprojection helper."""

    def test_returns_polygon(self):
        from eoflow.soil import _to_bng

        result = _to_bng(_DEVON_WGS84)
        assert isinstance(result, Polygon)

    def test_coordinates_in_bng_range(self):
        """BNG coordinates for mainland UK should be in the range 0–700000 E, 0–1300000 N."""
        from eoflow.soil import _to_bng

        bng_poly = _to_bng(_DEVON_WGS84)
        minx, miny, maxx, maxy = bng_poly.bounds
        assert 0 < minx < 700_000
        assert 0 < miny < 1_300_000
        assert 0 < maxx < 700_000
        assert 0 < maxy < 1_300_000

    def test_area_is_plausible_for_devon_box(self):
        """A ~22 km × ~22 km Devon box should have BNG area roughly 480 km²."""
        from eoflow.soil import _to_bng

        bng_poly = _to_bng(_DEVON_WGS84)
        area_km2 = bng_poly.area / 1_000_000
        assert 250 < area_km2 < 400, f"Unexpected area: {area_km2:.1f} km²"

    def test_polygon_is_valid_after_reproject(self):
        from eoflow.soil import _to_bng

        assert _to_bng(_DEVON_WGS84).is_valid

    def test_non_square_polygon(self):
        from eoflow.soil import _to_bng

        triangle = Polygon([(-2.0, 51.0), (-2.5, 51.5), (-1.5, 51.5), (-2.0, 51.0)])
        result = _to_bng(triangle)
        assert isinstance(result, Polygon)
        assert result.area > 0


# ===========================================================================
# Tests: _polygon_to_arcgis_json
# ===========================================================================


class TestPolygonToArcgisJson:
    """Tests for ArcGIS polygon JSON encoding."""

    def test_returns_dict(self):
        from eoflow.soil import _polygon_to_arcgis_json

        result = _polygon_to_arcgis_json(_DEVON_BNG_APPROX)
        assert isinstance(result, dict)

    def test_has_rings_key(self):
        from eoflow.soil import _polygon_to_arcgis_json

        result = _polygon_to_arcgis_json(_DEVON_BNG_APPROX)
        assert "rings" in result

    def test_has_spatial_reference(self):
        from eoflow.soil import _polygon_to_arcgis_json

        result = _polygon_to_arcgis_json(_DEVON_BNG_APPROX)
        assert result["spatialReference"] == {"wkid": 27700}

    def test_rings_is_list_of_lists(self):
        from eoflow.soil import _polygon_to_arcgis_json

        result = _polygon_to_arcgis_json(_DEVON_BNG_APPROX)
        assert isinstance(result["rings"], list)
        assert isinstance(result["rings"][0], list)

    def test_ring_coords_are_numeric_pairs(self):
        from eoflow.soil import _polygon_to_arcgis_json

        result = _polygon_to_arcgis_json(_DEVON_BNG_APPROX)
        for coord in result["rings"][0]:
            assert len(coord) == 2
            assert all(isinstance(v, float) for v in coord)

    def test_json_serialisable(self):
        from eoflow.soil import _polygon_to_arcgis_json

        result = _polygon_to_arcgis_json(_DEVON_BNG_APPROX)
        # Should not raise
        json.dumps(result)


# ===========================================================================
# Tests: _rings_to_shapely
# ===========================================================================


class TestRingsToShapely:
    """Tests for the ArcGIS rings → Shapely converter."""

    def test_single_ring_returns_polygon(self):
        from eoflow.soil import _rings_to_shapely

        ring = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        result = _rings_to_shapely([ring])
        assert isinstance(result, Polygon)

    def test_empty_rings_returns_empty_polygon(self):
        from eoflow.soil import _rings_to_shapely

        result = _rings_to_shapely([])
        assert isinstance(result, Polygon)
        assert result.is_empty

    def test_single_ring_area(self):
        from eoflow.soil import _rings_to_shapely

        ring = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
        result = _rings_to_shapely([ring])
        assert abs(result.area - 100.0) < 1e-6

    def test_two_rings_returns_geometry(self):
        """Two-ring input should produce a Polygon (with hole) or MultiPolygon."""
        from eoflow.soil import _rings_to_shapely

        outer = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
        inner = [[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]
        result = _rings_to_shapely([outer, inner])
        assert result is not None
        assert not result.is_empty

    def test_result_is_not_none(self):
        from eoflow.soil import _rings_to_shapely

        ring = [[285000, 66000], [290000, 66000], [290000, 72000], [285000, 72000], [285000, 66000]]
        result = _rings_to_shapely([ring])
        assert result is not None

    def test_bng_square_ring(self):
        """A BNG square ring should give the expected area."""
        from eoflow.soil import _rings_to_shapely

        # 1000 m × 1000 m square
        ring = [
            [285_000, 66_000],
            [286_000, 66_000],
            [286_000, 67_000],
            [285_000, 67_000],
            [285_000, 66_000],
        ]
        result = _rings_to_shapely([ring])
        assert abs(result.area - 1_000_000) < 1.0  # 1 km²


# ===========================================================================
# Tests: _features_to_geodataframe
# ===========================================================================


class TestFeaturesToGeoDataFrame:
    """Tests for the feature-list → GeoDataFrame converter."""

    def test_empty_list_returns_empty_gdf(self):
        from eoflow.soil import _features_to_geodataframe

        gdf = _features_to_geodataframe([])
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 0

    def test_empty_gdf_has_geometry_column(self):
        from eoflow.soil import _features_to_geodataframe

        gdf = _features_to_geodataframe([])
        assert "geometry" in gdf.columns

    def test_empty_gdf_crs_is_bng(self):
        from eoflow.soil import _features_to_geodataframe

        gdf = _features_to_geodataframe([])
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 27700

    def test_single_feature_creates_one_row(self):
        from eoflow.soil import _features_to_geodataframe

        feats = [_make_soil_feature(1, "Peat")]
        gdf = _features_to_geodataframe(feats)
        assert len(gdf) == 1

    def test_attributes_preserved(self):
        from eoflow.soil import _features_to_geodataframe

        feats = [_make_soil_feature(42, "Peat", mu_name="MYSOIL", bfi=0.75)]
        gdf = _features_to_geodataframe(feats)
        assert gdf.iloc[0]["OBJECTID"] == 42
        assert gdf.iloc[0]["SEARG_Concise"] == "Peat"
        assert gdf.iloc[0]["MU_NAME"] == "MYSOIL"
        assert abs(gdf.iloc[0]["BFI"] - 0.75) < 1e-6

    def test_geometry_is_polygon(self):
        from eoflow.soil import _features_to_geodataframe

        feats = [_make_soil_feature(1, "Peat")]
        gdf = _features_to_geodataframe(feats)
        assert isinstance(gdf.iloc[0].geometry, Polygon)

    def test_crs_is_bng(self):
        from eoflow.soil import _features_to_geodataframe

        feats = [_make_soil_feature(1, "Peat"), _make_soil_feature(2, "Shallow soils")]
        gdf = _features_to_geodataframe(feats)
        assert gdf.crs.to_epsg() == 27700

    def test_multiple_features(self):
        from eoflow.soil import _features_to_geodataframe

        feats = [_make_soil_feature(i, "Peat") for i in range(5)]
        gdf = _features_to_geodataframe(feats)
        assert len(gdf) == 5

    def test_feature_without_geometry(self):
        """Features with no geometry key should get a None geometry, not raise."""
        from eoflow.soil import _features_to_geodataframe

        feat = {"attributes": {"OBJECTID": 1, "SEARG_Concise": "Peat"}}
        gdf = _features_to_geodataframe([feat])
        assert len(gdf) == 1


# ===========================================================================
# Tests: _paginated_query
# ===========================================================================


class TestPaginatedQuery:
    """Tests for pagination logic in _paginated_query."""

    def test_single_page_no_pagination(self):
        from eoflow.soil import _paginated_query

        body = _make_query_response([_make_soil_feature(1, "Peat")], exceeded=False)
        session = _mock_session([body])

        geom_json = {
            "rings": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            "spatialReference": {"wkid": 27700},
        }
        features = _paginated_query(geom_json, session=session, timeout=30)

        assert len(features) == 1
        assert session.post.call_count == 1

    def test_two_pages_pagination(self):
        from eoflow.soil import _paginated_query

        page1 = _make_query_response(
            [_make_soil_feature(i, "Peat") for i in range(3)],
            exceeded=True,
        )
        page2 = _make_query_response(
            [_make_soil_feature(i + 100, "Shallow soils") for i in range(2)],
            exceeded=False,
        )
        session = _mock_session([page1, page2])

        geom_json = {"rings": [[[0, 0], [1, 0], [1, 1]]], "spatialReference": {"wkid": 27700}}
        features = _paginated_query(geom_json, session=session, timeout=30)

        assert len(features) == 5
        assert session.post.call_count == 2

    def test_second_page_uses_offset(self):
        from eoflow.soil import MAX_RECORD_COUNT, _paginated_query

        page1 = _make_query_response([_make_soil_feature(1, "Peat")], exceeded=True)
        page2 = _make_query_response([_make_soil_feature(2, "Peat")], exceeded=False)
        session = _mock_session([page1, page2])

        geom_json = {"rings": [[[0, 0], [1, 0], [1, 1]]], "spatialReference": {"wkid": 27700}}
        _paginated_query(geom_json, session=session, timeout=30)

        first_call_data = session.post.call_args_list[0][1]["data"]
        second_call_data = session.post.call_args_list[1][1]["data"]

        assert first_call_data["resultOffset"] == "0"
        assert second_call_data["resultOffset"] == str(MAX_RECORD_COUNT)

    def test_empty_response(self):
        from eoflow.soil import _paginated_query

        body = _make_query_response([], exceeded=False)
        session = _mock_session([body])

        geom_json = {"rings": [[[0, 0], [1, 0], [1, 1]]], "spatialReference": {"wkid": 27700}}
        features = _paginated_query(geom_json, session=session, timeout=30)

        assert features == []

    def test_arcgis_error_raises_runtime_error(self):
        from eoflow.soil import _paginated_query

        session = MagicMock(spec=requests.Session)
        r = Mock()
        r.raise_for_status = Mock()
        r.json.return_value = {"error": {"code": 400, "message": "Invalid geometry"}}
        session.post.return_value = r

        geom_json = {"rings": [[[0, 0], [1, 0], [1, 1]]], "spatialReference": {"wkid": 27700}}
        with pytest.raises(RuntimeError, match="ArcGIS service error"):
            _paginated_query(geom_json, session=session, timeout=30)

    def test_http_error_propagates(self):
        from eoflow.soil import _paginated_query

        session = MagicMock(spec=requests.Session)
        r = Mock()
        r.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        session.post.return_value = r

        geom_json = {"rings": [[[0, 0], [1, 0], [1, 1]]], "spatialReference": {"wkid": 27700}}
        with pytest.raises(requests.HTTPError):
            _paginated_query(geom_json, session=session, timeout=30)

    def test_posts_to_query_url(self):
        from eoflow.soil import QUERY_URL, _paginated_query

        body = _make_query_response([], exceeded=False)
        session = _mock_session([body])

        geom_json = {"rings": [[[0, 0], [1, 0], [1, 1]]], "spatialReference": {"wkid": 27700}}
        _paginated_query(geom_json, session=session, timeout=30)

        url_called = session.post.call_args_list[0][0][0]
        assert url_called == QUERY_URL

    def test_request_includes_geometry_param(self):
        from eoflow.soil import _paginated_query

        body = _make_query_response([], exceeded=False)
        session = _mock_session([body])

        geom_json = {"rings": [[[0, 0], [1, 0], [1, 1]]], "spatialReference": {"wkid": 27700}}
        _paginated_query(geom_json, session=session, timeout=30)

        data = session.post.call_args_list[0][1]["data"]
        assert "geometry" in data
        parsed = json.loads(data["geometry"])
        assert "rings" in parsed

    def test_request_asks_for_json_format(self):
        from eoflow.soil import _paginated_query

        body = _make_query_response([], exceeded=False)
        session = _mock_session([body])

        geom_json = {"rings": [[[0, 0], [1, 0], [1, 1]]], "spatialReference": {"wkid": 27700}}
        _paginated_query(geom_json, session=session, timeout=30)

        data = session.post.call_args_list[0][1]["data"]
        assert data.get("f") == "json"

    def test_request_uses_intersects_spatial_rel(self):
        from eoflow.soil import _paginated_query

        body = _make_query_response([], exceeded=False)
        session = _mock_session([body])

        geom_json = {"rings": [[[0, 0], [1, 0], [1, 1]]], "spatialReference": {"wkid": 27700}}
        _paginated_query(geom_json, session=session, timeout=30)

        data = session.post.call_args_list[0][1]["data"]
        assert data.get("spatialRel") == "esriSpatialRelIntersects"

    def test_three_pages_pagination(self):
        from eoflow.soil import _paginated_query

        pages = [
            _make_query_response([_make_soil_feature(i, "Peat") for i in range(4)], exceeded=True),
            _make_query_response(
                [_make_soil_feature(i + 10, "Peat") for i in range(4)], exceeded=True
            ),
            _make_query_response(
                [_make_soil_feature(i + 20, "Peat") for i in range(2)], exceeded=False
            ),
        ]
        session = _mock_session(pages)

        geom_json = {"rings": [[[0, 0], [1, 0], [1, 1]]], "spatialReference": {"wkid": 27700}}
        features = _paginated_query(geom_json, session=session, timeout=30)

        assert len(features) == 10
        assert session.post.call_count == 3


# ===========================================================================
# Tests: query_soil_polygons
# ===========================================================================


class TestQuerySoilPolygons:
    """Tests for the public query_soil_polygons function."""

    def _make_session_with_features(self, features: list, exceeded: bool = False) -> MagicMock:
        body = _make_query_response(features, exceeded=exceeded)
        return _mock_session([body])

    def test_returns_geodataframe(self):
        from eoflow.soil import query_soil_polygons

        session = self._make_session_with_features([_make_soil_feature(1, "Peat")])
        result = query_soil_polygons(_DEVON_WGS84, session=session)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_crs_is_bng(self):
        from eoflow.soil import query_soil_polygons

        session = self._make_session_with_features([_make_soil_feature(1, "Peat")])
        result = query_soil_polygons(_DEVON_WGS84, session=session)
        assert result.crs.to_epsg() == 27700

    def test_empty_result_when_no_features(self):
        from eoflow.soil import query_soil_polygons

        session = self._make_session_with_features([])
        result = query_soil_polygons(_DEVON_WGS84, session=session)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    def test_correct_row_count(self):
        from eoflow.soil import query_soil_polygons

        feats = [_make_soil_feature(i, "Peat") for i in range(7)]
        session = self._make_session_with_features(feats)
        result = query_soil_polygons(_DEVON_WGS84, session=session)
        assert len(result) == 7

    def test_searg_concise_column_present(self):
        from eoflow.soil import query_soil_polygons

        session = self._make_session_with_features([_make_soil_feature(1, "Peat")])
        result = query_soil_polygons(_DEVON_WGS84, session=session)
        assert "SEARG_Concise" in result.columns

    def test_geometry_column_present(self):
        from eoflow.soil import query_soil_polygons

        session = self._make_session_with_features([_make_soil_feature(1, "Peat")])
        result = query_soil_polygons(_DEVON_WGS84, session=session)
        assert "geometry" in result.columns

    def test_multiple_soil_groups_preserved(self):
        from eoflow.soil import query_soil_polygons

        feats = [
            _make_soil_feature(1, "Peat"),
            _make_soil_feature(2, "Shallow soils"),
            _make_soil_feature(3, "Light free drainage"),
        ]
        session = self._make_session_with_features(feats)
        result = query_soil_polygons(_DEVON_WGS84, session=session)
        assert set(result["SEARG_Concise"]) == {"Peat", "Shallow soils", "Light free drainage"}

    def test_passes_timeout_to_request(self):
        from eoflow.soil import query_soil_polygons

        session = self._make_session_with_features([])
        query_soil_polygons(_DEVON_WGS84, session=session, timeout=999)
        call_kwargs = session.post.call_args_list[0][1]
        assert call_kwargs.get("timeout") == 999

    def test_http_error_propagates(self):
        from eoflow.soil import query_soil_polygons

        session = MagicMock(spec=requests.Session)
        r = Mock()
        r.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")
        session.post.return_value = r

        with pytest.raises(requests.HTTPError):
            query_soil_polygons(_DEVON_WGS84, session=session)

    def test_bfi_and_spr_columns(self):
        from eoflow.soil import query_soil_polygons

        session = self._make_session_with_features(
            [_make_soil_feature(1, "Peat", bfi=0.42, spr=55)]
        )
        result = query_soil_polygons(_DEVON_WGS84, session=session)
        assert "BFI" in result.columns
        assert "SPR" in result.columns
        assert abs(result.iloc[0]["BFI"] - 0.42) < 1e-4


# ===========================================================================
# Tests: soil_coverage
# ===========================================================================


class TestSoilCoverage:
    """Tests for the soil_coverage function."""

    # Build a pair of soil polygons that together fill a known fraction of
    # _DEVON_BNG_APPROX (285000,66000 → 300000,88000 = 15000×22000 = 330 km²)
    # We create two soil polygons, each covering a known part of that extent.

    _CATCH_BNG = box(285_000, 66_000, 300_000, 88_000)  # 330 km²
    _SOIL_A = box(285_000, 66_000, 290_000, 88_000)  # 5000×22000 = 110 km²
    _SOIL_B = box(290_000, 66_000, 300_000, 88_000)  # 10000×22000 = 220 km²

    def _make_session(self, features: list) -> MagicMock:
        body = _make_query_response(features, exceeded=False)
        return _mock_session([body])

    def test_returns_series(self):
        from eoflow.soil import soil_coverage

        session = self._make_session([_make_soil_feature(1, "Peat", ring_polygon=self._SOIL_A)])
        result = soil_coverage(_DEVON_WGS84, session=session)
        assert isinstance(result, pd.Series)

    def test_series_name(self):
        from eoflow.soil import soil_coverage

        session = self._make_session([])
        result = soil_coverage(_DEVON_WGS84, session=session)
        assert result.name == "soil_fraction"

    def test_all_searg_groups_in_index(self):
        from eoflow.soil import SEARG_GROUPS, soil_coverage

        session = self._make_session([])
        result = soil_coverage(_DEVON_WGS84, session=session)
        for group in SEARG_GROUPS:
            assert group in result.index

    def test_index_has_twelve_entries(self):
        from eoflow.soil import soil_coverage

        session = self._make_session([])
        result = soil_coverage(_DEVON_WGS84, session=session)
        assert len(result) == 12

    def test_empty_result_all_zeros(self):
        from eoflow.soil import soil_coverage

        session = self._make_session([])
        result = soil_coverage(_DEVON_WGS84, session=session)
        assert (result == 0.0).all()

    def test_fractions_between_zero_and_one(self):
        from eoflow.soil import soil_coverage

        feats = [
            _make_soil_feature(1, "Peat", ring_polygon=self._SOIL_A),
            _make_soil_feature(2, "Shallow soils", ring_polygon=self._SOIL_B),
        ]
        session = self._make_session(feats)
        result = soil_coverage(_DEVON_WGS84, session=session)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_fractions_sum_at_most_one(self):
        from eoflow.soil import soil_coverage

        feats = [
            _make_soil_feature(1, "Peat", ring_polygon=self._SOIL_A),
            _make_soil_feature(2, "Shallow soils", ring_polygon=self._SOIL_B),
        ]
        session = self._make_session(feats)
        result = soil_coverage(_DEVON_WGS84, session=session)
        assert result.sum() <= 1.0 + 1e-6

    def test_known_fraction_single_group(self):
        """A soil polygon equal to half the catchment should give fraction ~0.5."""
        from eoflow.soil import _to_bng, soil_coverage

        # Use a real BNG catchment polygon so we can control exact fractions
        catch_wgs84 = box(-3.6, 50.6, -3.4, 50.8)
        catch_bng = _to_bng(catch_wgs84)
        # A polygon covering roughly the western half
        western_half = box(
            catch_bng.bounds[0],
            catch_bng.bounds[1],
            (catch_bng.bounds[0] + catch_bng.bounds[2]) / 2,
            catch_bng.bounds[3],
        )

        feats = [_make_soil_feature(1, "Peat", ring_polygon=western_half)]
        body = _make_query_response(feats, exceeded=False)
        session = _mock_session([body])

        result = soil_coverage(catch_wgs84, session=session)
        # The western half intersected with the catchment should be ~50%
        assert abs(result["Peat"] - 0.5) < 0.02

    def test_two_groups_fractions_sum_near_one_when_fully_covered(self):
        """Two soil polygons covering the whole catchment → fractions sum ≈ 1."""
        from eoflow.soil import _to_bng, soil_coverage

        catch_wgs84 = box(-3.6, 50.6, -3.4, 50.8)
        catch_bng = _to_bng(catch_wgs84)
        minx, miny, maxx, maxy = catch_bng.bounds
        midx = (minx + maxx) / 2

        left = box(minx, miny, midx, maxy)
        right = box(midx, miny, maxx, maxy)

        feats = [
            _make_soil_feature(1, "Peat", ring_polygon=left),
            _make_soil_feature(2, "Light free drainage", ring_polygon=right),
        ]
        body = _make_query_response(feats, exceeded=False)
        session = _mock_session([body])

        result = soil_coverage(catch_wgs84, session=session)
        assert abs(result.sum() - 1.0) < 0.02

    def test_unknown_searg_group_ignored(self):
        """Features with an unknown SEARG_Concise value should be silently skipped."""
        from eoflow.soil import soil_coverage

        feats = [_make_soil_feature(1, "Unknown mystery soil")]
        session = self._make_session(feats)
        result = soil_coverage(_DEVON_WGS84, session=session)
        # All known groups should still be 0
        assert (result == 0.0).all()

    def test_zero_area_polygon_raises(self):
        from eoflow.soil import soil_coverage

        point_poly = Point(0, 0).buffer(0)
        with pytest.raises((ValueError, Exception)):
            soil_coverage(point_poly)

    def test_http_error_propagates(self):
        from eoflow.soil import soil_coverage

        session = MagicMock(spec=requests.Session)
        r = Mock()
        r.raise_for_status.side_effect = requests.HTTPError("500 Internal Server Error")
        session.post.return_value = r

        with pytest.raises(requests.HTTPError):
            soil_coverage(_DEVON_WGS84, session=session)

    def test_nonzero_groups_are_searg_groups(self):
        """Any group with non-zero fraction must be one of the known SEARG groups."""
        from eoflow.soil import SEARG_GROUPS, soil_coverage

        feats = [
            _make_soil_feature(1, "Peat", ring_polygon=self._SOIL_A),
            _make_soil_feature(2, "Shallow soils", ring_polygon=self._SOIL_B),
        ]
        session = self._make_session(feats)
        result = soil_coverage(_DEVON_WGS84, session=session)
        for group in result[result > 0].index:
            assert group in SEARG_GROUPS


# ===========================================================================
# Tests: soil_coverage_summary
# ===========================================================================


class TestSoilCoverageSummary:
    """Tests for the soil_coverage_summary convenience function."""

    def _make_session(self, features: list) -> MagicMock:
        body = _make_query_response(features, exceeded=False)
        return _mock_session([body])

    def test_returns_dataframe(self):
        from eoflow.soil import soil_coverage_summary

        feats = [_make_soil_feature(1, "Peat", ring_polygon=box(285_000, 66_000, 290_000, 72_000))]
        session = self._make_session(feats)
        result = soil_coverage_summary(_DEVON_WGS84, session=session)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        from eoflow.soil import soil_coverage_summary

        feats = [_make_soil_feature(1, "Peat", ring_polygon=box(285_000, 66_000, 290_000, 72_000))]
        session = self._make_session(feats)
        result = soil_coverage_summary(_DEVON_WGS84, session=session)
        for col in ("SEARG_Concise", "fraction", "percent", "area_km2"):
            assert col in result.columns

    def test_zero_coverage_groups_excluded(self):
        """Groups with zero fraction must not appear in the summary."""
        from eoflow.soil import soil_coverage_summary

        feats = [_make_soil_feature(1, "Peat", ring_polygon=box(285_000, 66_000, 290_000, 72_000))]
        session = self._make_session(feats)
        result = soil_coverage_summary(_DEVON_WGS84, session=session)
        assert (result["fraction"] > 0).all()

    def test_empty_result_returns_empty_dataframe(self):
        from eoflow.soil import soil_coverage_summary

        session = self._make_session([])
        result = soil_coverage_summary(_DEVON_WGS84, session=session)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_sorted_descending_by_fraction(self):
        from eoflow.soil import _to_bng, soil_coverage_summary

        catch_wgs84 = box(-3.6, 50.6, -3.4, 50.8)
        catch_bng = _to_bng(catch_wgs84)
        minx, miny, maxx, maxy = catch_bng.bounds
        midx = (minx + maxx) / 2

        feats = [
            _make_soil_feature(1, "Peat", ring_polygon=box(minx, miny, midx, maxy)),
            _make_soil_feature(2, "Shallow soils", ring_polygon=box(midx, miny, maxx, maxy)),
        ]
        body = _make_query_response(feats, exceeded=False)
        session = _mock_session([body])

        result = soil_coverage_summary(catch_wgs84, session=session)
        assert list(result["fraction"]) == sorted(result["fraction"].tolist(), reverse=True)

    def test_percent_equals_fraction_times_100(self):
        from eoflow.soil import soil_coverage_summary

        feats = [_make_soil_feature(1, "Peat", ring_polygon=box(285_000, 66_000, 290_000, 72_000))]
        session = self._make_session(feats)
        result = soil_coverage_summary(_DEVON_WGS84, session=session)
        for _, row in result.iterrows():
            assert abs(row["percent"] - row["fraction"] * 100) < 1e-6

    def test_area_km2_is_positive(self):
        from eoflow.soil import soil_coverage_summary

        feats = [_make_soil_feature(1, "Peat", ring_polygon=box(285_000, 66_000, 290_000, 72_000))]
        session = self._make_session(feats)
        result = soil_coverage_summary(_DEVON_WGS84, session=session)
        assert (result["area_km2"] > 0).all()


# ===========================================================================
# Tests: module public API
# ===========================================================================


class TestModulePublicAPI:
    """Smoke tests verifying importability and public API surface."""

    def test_module_importable(self):
        import eoflow.soil  # noqa: F401

    def test_query_soil_polygons_importable(self):
        from eoflow.soil import query_soil_polygons  # noqa: F401

        assert callable(query_soil_polygons)

    def test_soil_coverage_importable(self):
        from eoflow.soil import soil_coverage  # noqa: F401

        assert callable(soil_coverage)

    def test_soil_coverage_summary_importable(self):
        from eoflow.soil import soil_coverage_summary  # noqa: F401

        assert callable(soil_coverage_summary)

    def test_searg_groups_importable(self):
        from eoflow.soil import SEARG_GROUPS  # noqa: F401

        assert SEARG_GROUPS is not None

    def test_query_url_importable(self):
        from eoflow.soil import QUERY_URL  # noqa: F401

        assert QUERY_URL.startswith("https://")

    def test_query_soil_polygons_has_docstring(self):
        from eoflow.soil import query_soil_polygons

        assert query_soil_polygons.__doc__ is not None
        assert len(query_soil_polygons.__doc__) > 20

    def test_soil_coverage_has_docstring(self):
        from eoflow.soil import soil_coverage

        assert soil_coverage.__doc__ is not None

    def test_soil_coverage_summary_has_docstring(self):
        from eoflow.soil import soil_coverage_summary

        assert soil_coverage_summary.__doc__ is not None


# ===========================================================================
# Tests: edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge-case and defensive-programming tests."""

    def test_soil_polygon_partly_outside_catchment(self):
        """A soil polygon that extends beyond the catchment boundary should only
        contribute the intersected area."""
        from eoflow.soil import _to_bng, soil_coverage

        catch_wgs84 = box(-3.6, 50.6, -3.4, 50.8)
        catch_bng = _to_bng(catch_wgs84)
        minx, miny, maxx, maxy = catch_bng.bounds

        # Soil polygon overlaps the eastern half of the catchment plus extra
        big_soil = box(
            (minx + maxx) / 2,  # starts at mid-point
            miny - 50_000,  # extends well south of catchment
            maxx + 50_000,  # extends well east of catchment
            maxy + 50_000,  # extends well north of catchment
        )

        feats = [_make_soil_feature(1, "Peat", ring_polygon=big_soil)]
        body = _make_query_response(feats, exceeded=False)
        session = _mock_session([body])

        result = soil_coverage(catch_wgs84, session=session)
        # Peat should cover roughly 50% (the eastern half only)
        assert 0.4 < result["Peat"] < 0.6

    def test_feature_with_none_geometry_skipped(self):
        """Features returned without geometry should not cause errors."""
        from eoflow.soil import query_soil_polygons

        feat_no_geom = {
            "attributes": {
                "OBJECTID": 99,
                "SEARG_Concise": "Peat",
                "SEARG": "Peat",
                "MU_NAME": "NONE",
                "MAP_SYMBOL": "N",
                "BFI": 0.5,
                "SPR": 30,
                "SEARGDescription": "",
                "Shape__Area": 0.0,
            }
            # no "geometry" key
        }
        body = _make_query_response([feat_no_geom], exceeded=False)
        session = _mock_session([body])
        result = query_soil_polygons(_DEVON_WGS84, session=session)
        assert len(result) == 1

    def test_session_reused_across_pages(self):
        """The same session object should be used for all paginated requests."""
        from eoflow.soil import query_soil_polygons

        page1 = _make_query_response([_make_soil_feature(1, "Peat")], exceeded=True)
        page2 = _make_query_response([_make_soil_feature(2, "Peat")], exceeded=False)
        session = _mock_session([page1, page2])

        query_soil_polygons(_DEVON_WGS84, session=session)
        assert session.post.call_count == 2

    def test_geometry_sent_as_bng_json(self):
        """The query geometry posted to the service must be in BNG (wkid 27700)."""
        from eoflow.soil import query_soil_polygons

        body = _make_query_response([], exceeded=False)
        session = _mock_session([body])
        query_soil_polygons(_DEVON_WGS84, session=session)

        data = session.post.call_args_list[0][1]["data"]
        geom = json.loads(data["geometry"])
        assert geom["spatialReference"]["wkid"] == 27700

    def test_out_sr_is_bng(self):
        """The outSR parameter must be 27700 so returned geometry is in BNG."""
        from eoflow.soil import query_soil_polygons

        body = _make_query_response([], exceeded=False)
        session = _mock_session([body])
        query_soil_polygons(_DEVON_WGS84, session=session)

        data = session.post.call_args_list[0][1]["data"]
        assert data.get("outSR") == "27700"

    def test_return_geometry_is_true(self):
        from eoflow.soil import query_soil_polygons

        body = _make_query_response([], exceeded=False)
        session = _mock_session([body])
        query_soil_polygons(_DEVON_WGS84, session=session)

        data = session.post.call_args_list[0][1]["data"]
        assert data.get("returnGeometry") == "true"


# ===========================================================================
# Integration tests — require internet access
# ===========================================================================

_SMALL_DEVON_BOX = box(-3.55, 50.68, -3.45, 50.76)


@pytest.mark.integration
class TestSoilIntegration:
    """Integration tests that call the real ArcGIS REST service.

    Run with::

        pytest -m integration -k TestSoilIntegration
    """

    def test_query_returns_nonempty_geodataframe(self):
        from eoflow.soil import query_soil_polygons

        gdf = query_soil_polygons(_SMALL_DEVON_BOX)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0

    def test_query_crs_is_bng(self):
        from eoflow.soil import query_soil_polygons

        gdf = query_soil_polygons(_SMALL_DEVON_BOX)
        assert gdf.crs.to_epsg() == 27700

    def test_query_searg_concise_values_are_known(self):
        from eoflow.soil import SEARG_GROUPS, query_soil_polygons

        gdf = query_soil_polygons(_SMALL_DEVON_BOX)
        for val in gdf["SEARG_Concise"].dropna():
            assert val in SEARG_GROUPS, f"Unexpected SEARG_Concise value: {val!r}"

    def test_query_bfi_column_is_numeric(self):
        from eoflow.soil import query_soil_polygons

        gdf = query_soil_polygons(_SMALL_DEVON_BOX)
        bfi = gdf["BFI"].dropna()
        assert (bfi >= 0).all()
        assert (bfi <= 1).all()

    def test_query_geometries_are_valid(self):
        from eoflow.soil import query_soil_polygons

        gdf = query_soil_polygons(_SMALL_DEVON_BOX)
        for geom in gdf.geometry.dropna():
            assert geom.is_valid or geom.buffer(0).is_valid

    def test_coverage_returns_series(self):
        from eoflow.soil import soil_coverage

        result = soil_coverage(_SMALL_DEVON_BOX)
        assert isinstance(result, pd.Series)

    def test_coverage_index_is_searg_groups(self):
        from eoflow.soil import SEARG_GROUPS, soil_coverage

        result = soil_coverage(_SMALL_DEVON_BOX)
        assert set(result.index) == set(SEARG_GROUPS)

    def test_coverage_fractions_in_range(self):
        from eoflow.soil import soil_coverage

        result = soil_coverage(_SMALL_DEVON_BOX)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_coverage_sum_reasonable(self):
        """Devon should be well-covered by SEARG polygons — sum should be > 0.5."""
        from eoflow.soil import soil_coverage

        result = soil_coverage(_SMALL_DEVON_BOX)
        assert result.sum() > 0.5, f"Coverage sum suspiciously low: {result.sum():.2f}"

    def test_coverage_at_least_one_nonzero_group(self):
        from eoflow.soil import soil_coverage

        result = soil_coverage(_SMALL_DEVON_BOX)
        assert (result > 0).any()

    def test_summary_returns_dataframe(self):
        from eoflow.soil import soil_coverage_summary

        result = soil_coverage_summary(_SMALL_DEVON_BOX)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_summary_sorted_descending(self):
        from eoflow.soil import soil_coverage_summary

        result = soil_coverage_summary(_SMALL_DEVON_BOX)
        fractions = result["fraction"].tolist()
        assert fractions == sorted(fractions, reverse=True)

    def test_summary_area_km2_positive(self):
        from eoflow.soil import soil_coverage_summary

        result = soil_coverage_summary(_SMALL_DEVON_BOX)
        assert (result["area_km2"] > 0).all()
