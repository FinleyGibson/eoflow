"""
Tests for the CS (Citizen Scientist) Data API module.

Tests cover:
- API initialization and configuration
- ArcGIS FeatureServer request handling (with mocking)
- Data fetching and processing
- Bounding box queries
- Polygon filtering
- Layer metadata retrieval
- Unique value queries
- Error handling
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eoflow.cs import CSDataAPI, get_cs_data


class TestCSDataAPIInit:
    """Tests for API initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config values."""
        api = CSDataAPI()
        assert api.delay == 1.0
        assert api.timeout == 30
        assert api.max_records == 2000
        assert api.layer_id == 0

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        api = CSDataAPI(
            base_url="https://example.com/FeatureServer",
            layer_id=1,
            delay=0.5,
            timeout=60,
            max_records=1000,
        )
        assert api.base_url == "https://example.com/FeatureServer"
        assert api.layer_id == 1
        assert api.delay == 0.5
        assert api.timeout == 60
        assert api.max_records == 1000
        assert api.query_url == "https://example.com/FeatureServer/1/query"

    def test_init_creates_session(self):
        """Test that initialization creates a requests session."""
        api = CSDataAPI()
        assert hasattr(api, "session")
        assert api.session is not None


class TestMakeRequest:
    """Tests for _make_request method."""

    def test_make_request_success(self):
        """Test successful API request."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {
            "features": [
                {
                    "attributes": {"id": 1, "value": "test"},
                    "geometry": {"x": -4.0, "y": 50.5},
                }
            ]
        }

        with patch.object(api.session, "get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()

            result = api._make_request({"where": "1=1"})

            assert result == mock_response
            assert "features" in result
            assert len(result["features"]) == 1

    def test_make_request_empty_response(self):
        """Test request with empty features array."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {"features": []}

        with patch.object(api.session, "get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()

            result = api._make_request({"where": "1=1"})

            assert result == mock_response
            assert result["features"] == []

    def test_make_request_api_error(self):
        """Test handling of ArcGIS API errors."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        error_response = {
            "error": {
                "message": "Invalid query",
                "details": ["Field not found"],
            }
        }

        with patch.object(api.session, "get") as mock_get:
            mock_get.return_value.json.return_value = error_response
            mock_get.return_value.raise_for_status = Mock()

            with pytest.raises(ValueError, match="ArcGIS API error"):
                api._make_request({"where": "invalid"})

    def test_make_request_parameters(self):
        """Test that default parameters are correctly merged."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {"features": []}

        with patch.object(api.session, "get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()

            api._make_request({"where": "1=1", "outFields": "id,name"})

            # Check that the call was made with merged parameters
            call_args = mock_get.call_args
            params = call_args[1]["params"]

            assert params["f"] == "json"
            assert params["where"] == "1=1"
            assert params["outFields"] == "id,name"
            assert params["returnGeometry"] == "true"
            assert params["spatialRel"] == "esriSpatialRelIntersects"

    def test_make_request_timeout(self):
        """Test request timeout handling."""
        api = CSDataAPI(
            base_url="https://example.com/FeatureServer", delay=0, timeout=1
        )

        with patch.object(api.session, "get") as mock_get:
            mock_get.side_effect = Exception("Timeout")

            with pytest.raises(Exception, match="Timeout"):
                api._make_request({"where": "1=1"})


class TestGetData:
    """Tests for get_data method."""

    def test_get_data_basic(self):
        """Test basic data fetching."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {
            "features": [
                {
                    "attributes": {
                        "objectid": 1,
                        "site_name": "Test Site",
                        "value": 15.5,
                    },
                    "geometry": {"x": -4.0, "y": 50.5},
                }
            ]
        }

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            df = api.get_data(where="1=1")

            assert not df.empty
            assert len(df) == 1
            assert "objectid" in df.columns
            assert "site_name" in df.columns
            assert "longitude" in df.columns
            assert "latitude" in df.columns
            assert df.iloc[0]["longitude"] == -4.0
            assert df.iloc[0]["latitude"] == 50.5

    def test_get_data_with_date_filtering(self):
        """Test data fetching with date range."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {"features": []}

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            df = api.get_data(
                where="1=1",
                start_date="2024-01-01",
                end_date="2024-01-31",
                date_field="sample_date",
            )

            # Check that the WHERE clause includes date filtering
            call_args = mock_request.call_args[0][0]
            where_clause = call_args["where"]

            assert "sample_date" in where_clause
            assert ">=" in where_clause
            assert "<=" in where_clause

    def test_get_data_datetime_conversion(self):
        """Test conversion of timestamp fields to datetime."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        # ArcGIS uses milliseconds since epoch
        timestamp_ms = 1704067200000  # 2024-01-01 00:00:00

        mock_response = {
            "features": [
                {
                    "attributes": {
                        "objectid": 1,
                        "sample_date": timestamp_ms,
                        "value": 15.5,
                    },
                    "geometry": {"x": -4.0, "y": 50.5},
                }
            ]
        }

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            df = api.get_data()

            assert not df.empty
            assert "sample_date" in df.columns
            # Check if it's a datetime type
            assert pd.api.types.is_datetime64_any_dtype(df["sample_date"])

    def test_get_data_no_data_returned(self):
        """Test handling of empty response."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {"features": []}

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            df = api.get_data()

            assert df.empty

    def test_get_data_with_geometry_filter(self):
        """Test data fetching with geometry filter."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {"features": []}

        geometry = json.dumps({"xmin": -5, "ymin": 50, "xmax": -3, "ymax": 52})

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            df = api.get_data(
                geometry=geometry, geometry_type="esriGeometryEnvelope"
            )

            call_args = mock_request.call_args[0][0]
            assert "geometry" in call_args
            assert "geometryType" in call_args
            assert call_args["geometryType"] == "esriGeometryEnvelope"

    def test_get_data_polygon_geometry(self):
        """Test handling of polygon geometry in features."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {
            "features": [
                {
                    "attributes": {"objectid": 1, "name": "Polygon Site"},
                    "geometry": {
                        "rings": [[[-4.0, 50.0], [-4.0, 51.0], [-3.0, 51.0]]]
                    },
                }
            ]
        }

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            df = api.get_data()

            assert not df.empty
            assert "geometry_json" in df.columns
            # Check that geometry was stored as JSON string
            geom_data = json.loads(df.iloc[0]["geometry_json"])
            assert "rings" in geom_data


class TestGetDataByBBox:
    """Tests for get_data_by_bbox method."""

    def test_get_data_by_bbox_basic(self):
        """Test basic bounding box query."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {
            "features": [
                {
                    "attributes": {"objectid": 1, "value": 10.5},
                    "geometry": {"x": -4.0, "y": 50.5},
                }
            ]
        }

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            df = api.get_data_by_bbox(
                xmin=-5.0, ymin=50.0, xmax=-3.0, ymax=52.0
            )

            assert not df.empty
            call_args = mock_request.call_args[0][0]
            assert "geometry" in call_args

            # Parse the geometry parameter
            geom = json.loads(call_args["geometry"])
            assert geom["xmin"] == -5.0
            assert geom["ymin"] == 50.0
            assert geom["xmax"] == -3.0
            assert geom["ymax"] == 52.0
            assert geom["spatialReference"]["wkid"] == 4326

    def test_get_data_by_bbox_with_custom_srid(self):
        """Test bounding box query with custom spatial reference."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {"features": []}

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            df = api.get_data_by_bbox(
                xmin=-5.0,
                ymin=50.0,
                xmax=-3.0,
                ymax=52.0,
                spatial_reference=27700,  # British National Grid
            )

            call_args = mock_request.call_args[0][0]
            geom = json.loads(call_args["geometry"])
            assert geom["spatialReference"]["wkid"] == 27700

    def test_get_data_by_bbox_with_dates(self):
        """Test bounding box query with date filtering."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {"features": []}

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            df = api.get_data_by_bbox(
                xmin=-5.0,
                ymin=50.0,
                xmax=-3.0,
                ymax=52.0,
                start_date="2024-01-01",
                end_date="2024-12-31",
            )

            call_args = mock_request.call_args[0][0]
            assert "sample_date" in call_args["where"]


class TestFilterByPolygon:
    """Tests for filter_by_polygon method."""

    def test_filter_by_polygon_with_list(self):
        """Test polygon filtering with list of tuples."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        # Create sample data with points
        df = pd.DataFrame(
            {
                "objectid": [1, 2, 3],
                "latitude": [50.5, 51.0, 52.0],
                "longitude": [-4.0, -3.5, -3.0],
                "value": [10.5, 15.2, 12.8],
            }
        )

        # Polygon around the southwest
        polygon = [
            (-5.0, 50.0),
            (-5.0, 51.5),
            (-3.0, 51.5),
            (-3.0, 50.0),
            (-5.0, 50.0),
        ]

        filtered_df = api.filter_by_polygon(df, polygon)

        # Points 1 and 2 should be inside, point 3 outside
        assert len(filtered_df) == 2
        assert 1 in filtered_df["objectid"].values
        assert 2 in filtered_df["objectid"].values
        assert 3 not in filtered_df["objectid"].values

    def test_filter_by_polygon_empty_dataframe(self):
        """Test polygon filtering with empty DataFrame."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        df = pd.DataFrame()
        polygon = [(-5.0, 50.0), (-5.0, 51.0), (-3.0, 51.0), (-3.0, 50.0)]

        filtered_df = api.filter_by_polygon(df, polygon)

        assert filtered_df.empty

    def test_filter_by_polygon_missing_columns(self):
        """Test polygon filtering with missing lat/lon columns."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        df = pd.DataFrame({"objectid": [1, 2], "value": [10.5, 15.2]})
        polygon = [(-5.0, 50.0), (-5.0, 51.0), (-3.0, 51.0), (-3.0, 50.0)]

        filtered_df = api.filter_by_polygon(df, polygon)

        # Should return original DataFrame when columns are missing
        assert len(filtered_df) == len(df)

    def test_filter_by_polygon_custom_columns(self):
        """Test polygon filtering with custom column names."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        df = pd.DataFrame(
            {
                "objectid": [1, 2],
                "lat": [50.5, 52.0],
                "lon": [-4.0, -3.0],
                "value": [10.5, 12.8],
            }
        )

        polygon = [
            (-5.0, 50.0),
            (-5.0, 51.5),
            (-3.0, 51.5),
            (-3.0, 50.0),
            (-5.0, 50.0),
        ]

        filtered_df = api.filter_by_polygon(
            df, polygon, lat_col="lat", lon_col="lon"
        )

        assert len(filtered_df) == 1
        assert 1 in filtered_df["objectid"].values

    def test_filter_by_polygon_invalid_coordinates(self):
        """Test polygon filtering with invalid/missing coordinates."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        df = pd.DataFrame(
            {
                "objectid": [1, 2, 3],
                "latitude": [50.5, None, 51.0],
                "longitude": [-4.0, -3.5, None],
                "value": [10.5, 15.2, 12.8],
            }
        )

        polygon = [
            (-5.0, 50.0),
            (-5.0, 51.5),
            (-3.0, 51.5),
            (-3.0, 50.0),
            (-5.0, 50.0),
        ]

        filtered_df = api.filter_by_polygon(df, polygon)

        # Only point 1 should be included (2 and 3 have missing coords)
        assert len(filtered_df) == 1
        assert 1 in filtered_df["objectid"].values

    def test_filter_by_polygon_no_shapely(self):
        """Test error handling when shapely is not installed."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        df = pd.DataFrame(
            {
                "objectid": [1],
                "latitude": [50.5],
                "longitude": [-4.0],
            }
        )

        polygon = [(-5.0, 50.0), (-5.0, 51.0), (-3.0, 51.0), (-3.0, 50.0)]

        with patch.dict(
            "sys.modules", {"shapely": None, "shapely.geometry": None}
        ):
            with pytest.raises(ImportError, match="shapely is required"):
                api.filter_by_polygon(df, polygon)


class TestGetLayerInfo:
    """Tests for get_layer_info method."""

    def test_get_layer_info_success(self):
        """Test successful layer info retrieval."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_info = {
            "name": "CS_Water_Quality",
            "type": "Feature Layer",
            "geometryType": "esriGeometryPoint",
            "fields": [
                {"name": "objectid", "type": "esriFieldTypeOID"},
                {"name": "site_name", "type": "esriFieldTypeString"},
            ],
        }

        with patch.object(api.session, "get") as mock_get:
            mock_get.return_value.json.return_value = mock_info
            mock_get.return_value.raise_for_status = Mock()

            info = api.get_layer_info()

            assert info["name"] == "CS_Water_Quality"
            assert info["type"] == "Feature Layer"
            assert len(info["fields"]) == 2

    def test_get_layer_info_error(self):
        """Test layer info error handling."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        error_response = {"error": {"message": "Layer not found"}}

        with patch.object(api.session, "get") as mock_get:
            mock_get.return_value.json.return_value = error_response
            mock_get.return_value.raise_for_status = Mock()

            with pytest.raises(ValueError, match="ArcGIS API error"):
                api.get_layer_info()


class TestGetUniqueValues:
    """Tests for get_unique_values method."""

    def test_get_unique_values_success(self):
        """Test getting unique values from a field."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {
            "features": [
                {"attributes": {"site_type": "River"}},
                {"attributes": {"site_type": "Lake"}},
                {"attributes": {"site_type": "River"}},
                {"attributes": {"site_type": "Stream"}},
            ]
        }

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            unique_values = api.get_unique_values("site_type")

            assert len(unique_values) == 3
            assert "River" in unique_values
            assert "Lake" in unique_values
            assert "Stream" in unique_values

    def test_get_unique_values_empty_data(self):
        """Test getting unique values when no data exists."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {"features": []}

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            unique_values = api.get_unique_values("site_type")

            assert unique_values == []

    def test_get_unique_values_with_nulls(self):
        """Test getting unique values with null values."""
        api = CSDataAPI(base_url="https://example.com/FeatureServer", delay=0)

        mock_response = {
            "features": [
                {"attributes": {"site_type": "River"}},
                {"attributes": {"site_type": None}},
                {"attributes": {"site_type": "Lake"}},
            ]
        }

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = mock_response

            unique_values = api.get_unique_values("site_type")

            # Null values should be excluded
            assert len(unique_values) == 2
            assert None not in unique_values


class TestConvenienceFunction:
    """Tests for get_cs_data convenience function."""

    def test_get_cs_data_basic(self):
        """Test basic usage of convenience function."""
        mock_response = {
            "features": [
                {
                    "attributes": {"objectid": 1, "value": 15.5},
                    "geometry": {"x": -4.0, "y": 50.5},
                }
            ]
        }

        with patch("eoflow.cs.CSDataAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.get_data.return_value = pd.DataFrame(
                {
                    "objectid": [1],
                    "value": [15.5],
                    "longitude": [-4.0],
                    "latitude": [50.5],
                }
            )

            df = get_cs_data(
                base_url="https://example.com/FeatureServer",
                where="value > 10",
            )

            assert not df.empty
            assert len(df) == 1

    def test_get_cs_data_with_bbox(self):
        """Test convenience function with bounding box."""
        with patch("eoflow.cs.CSDataAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.get_data_by_bbox.return_value = pd.DataFrame(
                {"objectid": [1], "value": [15.5]}
            )

            df = get_cs_data(
                base_url="https://example.com/FeatureServer",
                bbox=(-5.0, 50.0, -3.0, 52.0),
            )

            # Verify bbox method was called
            mock_api.get_data_by_bbox.assert_called_once()
            assert not df.empty

    def test_get_cs_data_with_polygon_filtering(self):
        """Test convenience function with polygon filtering."""
        with patch("eoflow.cs.CSDataAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            initial_df = pd.DataFrame(
                {
                    "objectid": [1, 2],
                    "latitude": [50.5, 52.0],
                    "longitude": [-4.0, -3.0],
                }
            )
            filtered_df = pd.DataFrame(
                {"objectid": [1], "latitude": [50.5], "longitude": [-4.0]}
            )

            mock_api.get_data.return_value = initial_df
            mock_api.filter_by_polygon.return_value = filtered_df

            polygon = [
                (-5.0, 50.0),
                (-5.0, 51.0),
                (-3.0, 51.0),
                (-3.0, 50.0),
            ]

            df = get_cs_data(
                base_url="https://example.com/FeatureServer", polygon=polygon
            )

            # Verify polygon filtering was called
            mock_api.filter_by_polygon.assert_called_once()
            assert len(df) == 1

    def test_get_cs_data_with_dates(self):
        """Test convenience function with date filtering."""
        with patch("eoflow.cs.CSDataAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.get_data.return_value = pd.DataFrame()

            df = get_cs_data(
                base_url="https://example.com/FeatureServer",
                start_date="2024-01-01",
                end_date="2024-12-31",
            )

            # Verify get_data was called with date parameters
            call_kwargs = mock_api.get_data.call_args[1]
            assert call_kwargs["start_date"] == "2024-01-01"
            assert call_kwargs["end_date"] == "2024-12-31"


class TestIntegration:
    """Integration tests (skipped by default)."""

    @pytest.mark.skip(reason="Skipping real API test by default")
    def test_real_api_call(self):
        """
        Test with a real API endpoint.

        This test is skipped by default. To run it:
        pytest tests/test_cs.py::TestIntegration::test_real_api_call -v
        """
        # Example: Environment Agency Bathing Waters
        base_url = "https://environment.data.gov.uk/arcgis/rest/services/EA/SensitiveAreasBathingWaters/FeatureServer"

        api = CSDataAPI(base_url=base_url, layer_id=0)

        # Get layer info
        info = api.get_layer_info()
        assert "name" in info
        assert "fields" in info

        # Get some data with a WHERE clause
        df = api.get_data(where="1=1", verbose=True)

        # Basic checks
        assert isinstance(df, pd.DataFrame)
        print(f"\nRetrieved {len(df)} records from real API")
        if not df.empty:
            print(f"Columns: {list(df.columns)}")
