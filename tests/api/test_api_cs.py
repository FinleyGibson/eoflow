"""
Tests for CS and Combined Water Quality API endpoints in the FastAPI application.

Tests cover:
- CS water quality endpoint
- Combined EA + CS endpoint
- Request validation
- Error handling
- Data source tagging
"""

from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

from eoflow.api import app

client = TestClient(app)


class TestCSWaterQualityEndpoint:
    """Tests for /cs/water-quality endpoint."""

    def test_cs_water_quality_success(self):
        """Test successful CS water quality request."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "base_url": "https://services.arcgis.com/xxx/FeatureServer",
            "layer_id": 0,
            "where": "1=1",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "verbose": False,
        }

        with patch("eoflow.api.CSDataAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock get_data to return sample data
            sample_data = pd.DataFrame(
                {
                    "objectid": [1, 2],
                    "site_name": ["Site A", "Site B"],
                    "value": [15.5, 16.2],
                    "latitude": [50.5, 51.0],
                    "longitude": [-4.0, -3.5],
                }
            )
            mock_api.get_data.return_value = sample_data

            # Mock filter_by_polygon to return filtered data
            filtered_data = sample_data.iloc[[0]]
            mock_api.filter_by_polygon.return_value = filtered_data

            response = client.post("/cs/water-quality", json=request_data)

            # Verify get_data was called with correct parameters
            mock_api.get_data.assert_called_once_with(
                where="1=1",
                start_date="2024-01-01",
                end_date="2024-01-31",
                date_field="sample_date",
                verbose=False,
            )

            # Verify filter_by_polygon was called with correct polygon coordinates
            expected_coords = [
                (-4.5, 50.3),
                (-4.5, 51.2),
                (-3.0, 51.2),
                (-3.0, 50.3),
                (-4.5, 50.3),
            ]
            mock_api.filter_by_polygon.assert_called_once()
            call_args = mock_api.filter_by_polygon.call_args
            assert call_args[0][1] == expected_coords

            assert response.status_code == 200
            json_response = response.json()
            assert "total_records" in json_response
            assert "filtered_records" in json_response
            assert "data_source" in json_response
            assert "data" in json_response
            assert json_response["total_records"] == 2
            assert json_response["filtered_records"] == 1
            assert json_response["data_source"] == "CS"

    def test_cs_water_quality_empty_response(self):
        """Test CS water quality request with no data returned."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "base_url": "https://services.arcgis.com/xxx/FeatureServer",
            "layer_id": 0,
        }

        with patch("eoflow.api.CSDataAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock empty dataframe
            mock_api.get_data.return_value = pd.DataFrame()

            response = client.post("/cs/water-quality", json=request_data)

            assert response.status_code == 200
            json_response = response.json()
            assert json_response["total_records"] == 0
            assert json_response["filtered_records"] == 0
            assert json_response["data_source"] == "CS"
            assert json_response["data"] == []
            assert "message" in json_response

    def test_cs_water_quality_all_filtered_out(self):
        """Test when data is returned but all records are filtered out by polygon."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "base_url": "https://services.arcgis.com/xxx/FeatureServer",
        }

        with patch("eoflow.api.CSDataAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock get_data to return sample data
            sample_data = pd.DataFrame(
                {
                    "objectid": [1, 2],
                    "value": [15.5, 16.2],
                    "latitude": [50.5, 51.0],
                    "longitude": [-4.0, -3.5],
                }
            )
            mock_api.get_data.return_value = sample_data

            # Mock filter_by_polygon to return empty dataframe
            mock_api.filter_by_polygon.return_value = pd.DataFrame()

            response = client.post("/cs/water-quality", json=request_data)

            assert response.status_code == 200
            json_response = response.json()
            assert json_response["total_records"] == 2
            assert json_response["filtered_records"] == 0
            assert json_response["data"] == []

    def test_cs_water_quality_custom_parameters(self):
        """Test CS water quality with custom parameters."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "base_url": "https://services.arcgis.com/xxx/FeatureServer",
            "layer_id": 5,
            "where": "status = 'active'",
            "date_field": "observation_date",
            "verbose": True,
        }

        with patch("eoflow.api.CSDataAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.get_data.return_value = pd.DataFrame()

            response = client.post("/cs/water-quality", json=request_data)

            # Verify CSDataAPI was initialized with correct parameters
            mock_api_class.assert_called_once_with(
                base_url="https://services.arcgis.com/xxx/FeatureServer",
                layer_id=5,
            )

            # Verify get_data was called with custom parameters
            call_kwargs = mock_api.get_data.call_args[1]
            assert call_kwargs["where"] == "status = 'active'"
            assert call_kwargs["date_field"] == "observation_date"
            assert call_kwargs["verbose"] is True

            assert response.status_code == 200


class TestCombinedWaterQualityEndpoint:
    """Tests for /combined/water-quality endpoint."""

    def test_combined_water_quality_both_sources(self):
        """Test combined endpoint with both EA and CS data."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "ea_determinand": "0076",
            "ea_area": "environment_agency,SWX",
            "cs_base_url": "https://services.arcgis.com/xxx/FeatureServer",
            "cs_layer_id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        with (
            patch("eoflow.api.EAWaterQualityAPI") as mock_ea_class,
            patch("eoflow.api.CSDataAPI") as mock_cs_class,
        ):
            # Setup EA mock
            mock_ea = Mock()
            mock_ea_class.return_value = mock_ea
            ea_data = pd.DataFrame(
                {
                    "id": ["EA1", "EA2"],
                    "result": ["15.5", "16.2"],
                    "latitude": [50.5, 51.0],
                    "longitude": [-4.0, -3.5],
                }
            )
            mock_ea.get_data.return_value = ea_data
            mock_ea.filter_by_polygon.return_value = ea_data

            # Setup CS mock
            mock_cs = Mock()
            mock_cs_class.return_value = mock_cs
            cs_data = pd.DataFrame(
                {
                    "objectid": [1, 2],
                    "value": [14.8, 15.9],
                    "latitude": [50.6, 50.9],
                    "longitude": [-4.1, -3.6],
                }
            )
            mock_cs.get_data.return_value = cs_data
            mock_cs.filter_by_polygon.return_value = cs_data

            response = client.post("/combined/water-quality", json=request_data)

            assert response.status_code == 200
            json_response = response.json()

            # Check response structure
            assert "ea_records" in json_response
            assert "cs_records" in json_response
            assert "total_combined_records" in json_response
            assert "data" in json_response

            # Check counts
            assert json_response["ea_records"] == 2
            assert json_response["cs_records"] == 2
            assert json_response["total_combined_records"] == 4

            # Check data has source tags
            assert len(json_response["data"]) == 4
            sources = [record["data_source"] for record in json_response["data"]]
            assert sources.count("EA") == 2
            assert sources.count("CS") == 2

    def test_combined_water_quality_ea_only(self):
        """Test combined endpoint with only EA data."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "ea_determinand": "0076",
            "ea_area": "environment_agency,SWX",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_ea_class:
            mock_ea = Mock()
            mock_ea_class.return_value = mock_ea
            ea_data = pd.DataFrame(
                {
                    "id": ["EA1"],
                    "result": ["15.5"],
                    "latitude": [50.5],
                    "longitude": [-4.0],
                }
            )
            mock_ea.get_data.return_value = ea_data
            mock_ea.filter_by_polygon.return_value = ea_data

            response = client.post("/combined/water-quality", json=request_data)

            assert response.status_code == 200
            json_response = response.json()
            assert json_response["ea_records"] == 1
            assert json_response["cs_records"] == 0
            assert json_response["total_combined_records"] == 1

    def test_combined_water_quality_cs_only(self):
        """Test combined endpoint with only CS data."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "cs_base_url": "https://services.arcgis.com/xxx/FeatureServer",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        with patch("eoflow.api.CSDataAPI") as mock_cs_class:
            mock_cs = Mock()
            mock_cs_class.return_value = mock_cs
            cs_data = pd.DataFrame(
                {
                    "objectid": [1],
                    "value": [14.8],
                    "latitude": [50.6],
                    "longitude": [-4.1],
                }
            )
            mock_cs.get_data.return_value = cs_data
            mock_cs.filter_by_polygon.return_value = cs_data

            response = client.post("/combined/water-quality", json=request_data)

            assert response.status_code == 200
            json_response = response.json()
            assert json_response["ea_records"] == 0
            assert json_response["cs_records"] == 1
            assert json_response["total_combined_records"] == 1

    def test_combined_water_quality_no_data(self):
        """Test combined endpoint when no data is returned from either source."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "ea_determinand": "0076",
            "ea_area": "environment_agency,SWX",
            "cs_base_url": "https://services.arcgis.com/xxx/FeatureServer",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        with (
            patch("eoflow.api.EAWaterQualityAPI") as mock_ea_class,
            patch("eoflow.api.CSDataAPI") as mock_cs_class,
        ):
            mock_ea = Mock()
            mock_ea_class.return_value = mock_ea
            mock_ea.get_data.return_value = pd.DataFrame()

            mock_cs = Mock()
            mock_cs_class.return_value = mock_cs
            mock_cs.get_data.return_value = pd.DataFrame()

            response = client.post("/combined/water-quality", json=request_data)

            assert response.status_code == 200
            json_response = response.json()
            assert json_response["ea_records"] == 0
            assert json_response["cs_records"] == 0
            assert json_response["total_combined_records"] == 0
            assert json_response["data"] == []
            assert "message" in json_response

    def test_combined_water_quality_partial_failure(self):
        """Test combined endpoint when one source fails but the other succeeds."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "ea_determinand": "0076",
            "ea_area": "environment_agency,SWX",
            "cs_base_url": "https://services.arcgis.com/xxx/FeatureServer",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        with (
            patch("eoflow.api.EAWaterQualityAPI") as mock_ea_class,
            patch("eoflow.api.CSDataAPI") as mock_cs_class,
        ):
            # EA succeeds
            mock_ea = Mock()
            mock_ea_class.return_value = mock_ea
            ea_data = pd.DataFrame(
                {
                    "id": ["EA1"],
                    "result": ["15.5"],
                    "latitude": [50.5],
                    "longitude": [-4.0],
                }
            )
            mock_ea.get_data.return_value = ea_data
            mock_ea.filter_by_polygon.return_value = ea_data

            # CS fails
            mock_cs = Mock()
            mock_cs_class.return_value = mock_cs
            mock_cs.get_data.side_effect = Exception("API connection error")

            response = client.post("/combined/water-quality", json=request_data)

            assert response.status_code == 200
            json_response = response.json()

            # Should have EA data
            assert json_response["ea_records"] == 1
            assert json_response["cs_records"] == 0
            assert json_response["total_combined_records"] == 1

            # Should include error information
            assert "errors" in json_response
            assert any("CS API error" in err for err in json_response["errors"])

    def test_combined_water_quality_polygon_filtering(self):
        """Test that polygon filtering is applied to both sources."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "ea_determinand": "0076",
            "ea_area": "environment_agency,SWX",
            "cs_base_url": "https://services.arcgis.com/xxx/FeatureServer",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        with (
            patch("eoflow.api.EAWaterQualityAPI") as mock_ea_class,
            patch("eoflow.api.CSDataAPI") as mock_cs_class,
        ):
            mock_ea = Mock()
            mock_ea_class.return_value = mock_ea
            mock_ea.get_data.return_value = pd.DataFrame({"id": [1]})
            mock_ea.filter_by_polygon.return_value = pd.DataFrame({"id": [1]})

            mock_cs = Mock()
            mock_cs_class.return_value = mock_cs
            mock_cs.get_data.return_value = pd.DataFrame({"objectid": [1]})
            mock_cs.filter_by_polygon.return_value = pd.DataFrame({"objectid": [1]})

            response = client.post("/combined/water-quality", json=request_data)

            # Verify both sources had filter_by_polygon called
            expected_coords = [
                (-4.5, 50.3),
                (-4.5, 51.2),
                (-3.0, 51.2),
                (-3.0, 50.3),
                (-4.5, 50.3),
            ]

            ea_call_args = mock_ea.filter_by_polygon.call_args
            assert ea_call_args[0][1] == expected_coords

            cs_call_args = mock_cs.filter_by_polygon.call_args
            assert cs_call_args[0][1] == expected_coords

            assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling in CS and combined endpoints."""

    def test_cs_water_quality_value_error(self):
        """Test CS endpoint with invalid parameters."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "base_url": "https://services.arcgis.com/xxx/FeatureServer",
        }

        with patch("eoflow.api.CSDataAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.get_data.side_effect = ValueError("Invalid date format")

            response = client.post("/cs/water-quality", json=request_data)

            assert response.status_code == 400
            assert "Invalid request parameters" in response.json()["detail"]

    def test_cs_water_quality_import_error(self):
        """Test CS endpoint when required dependency is missing."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                    [-3.0, 51.2],
                    [-3.0, 50.3],
                    [-4.5, 50.3],
                ]
            },
            "base_url": "https://services.arcgis.com/xxx/FeatureServer",
        }

        with patch("eoflow.api.CSDataAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.get_data.return_value = pd.DataFrame({"id": [1]})
            mock_api.filter_by_polygon.side_effect = ImportError("shapely is required")

            response = client.post("/cs/water-quality", json=request_data)

            assert response.status_code == 500
            assert "Missing required dependency" in response.json()["detail"]
