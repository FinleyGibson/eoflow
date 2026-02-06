"""
Tests for EA Water Quality API endpoints in the FastAPI application.

Tests cover:
- EA water quality single determinand endpoint
- EA water quality multiple determinands endpoint
- Request validation
- Error handling
- Polygon filtering integration
"""

from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

from eoflow.api import app

client = TestClient(app)


class TestEAWaterQualityEndpoint:
    """Tests for /ea/water-quality endpoint."""

    def test_ea_water_quality_success(self):
        """Test successful EA water quality request."""

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
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
            "verbose": False,
        }

        # Mock the EA API
        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock get_data to return sample data
            sample_data = pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "result": ["15.5", "16.2"],
                    "phenomenonTime": [
                        "2024-01-01T10:00:00",
                        "2024-01-02T11:00:00",
                    ],
                    "sample.samplingPoint.latitude": ["50.5", "51.0"],
                    "sample.samplingPoint.longitude": ["-4.0", "-3.5"],
                }
            )
            mock_api.get_data.return_value = sample_data

            # Mock filter_by_polygon to return filtered data
            filtered_data = sample_data.iloc[[0]]  # Just first row
            mock_api.filter_by_polygon.return_value = filtered_data

            response = client.post("/ea/water-quality", json=request_data)

            # Verify get_data was called with correct parameters
            mock_api.get_data.assert_called_once_with(
                determinand="0076",
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="environment_agency,SWX",
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
            assert (
                call_args[0][1] == expected_coords
            )  # Second argument should be polygon coords

            assert response.status_code == 200
            json_response = response.json()
            assert "total_records" in json_response
            assert "filtered_records" in json_response
            assert "data" in json_response
            assert json_response["total_records"] == 2
            assert json_response["filtered_records"] == 1

    def test_ea_water_quality_verbose_true(self):
        """Test that verbose parameter is correctly passed through."""

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
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
            "verbose": True,  # Explicitly set to True
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            sample_data = pd.DataFrame(
                {
                    "id": ["1"],
                    "result": ["15.5"],
                    "phenomenonTime": ["2024-01-01T10:00:00"],
                    "sample.samplingPoint.latitude": ["50.5"],
                    "sample.samplingPoint.longitude": ["-4.0"],
                }
            )
            mock_api.get_data.return_value = sample_data
            mock_api.filter_by_polygon.return_value = sample_data

            response = client.post("/ea/water-quality", json=request_data)

            # Verify verbose=True was passed through
            mock_api.get_data.assert_called_once_with(
                determinand="0076",
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="environment_agency,SWX",
                verbose=True,
            )

            assert response.status_code == 200

    def test_ea_water_quality_all_filtered_out(self):
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
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock get_data to return sample data
            sample_data = pd.DataFrame(
                {
                    "id": ["1", "2"],
                    "result": ["15.5", "16.2"],
                    "phenomenonTime": [
                        "2024-01-01T10:00:00",
                        "2024-01-02T11:00:00",
                    ],
                    "sample.samplingPoint.latitude": ["50.5", "51.0"],
                    "sample.samplingPoint.longitude": ["-4.0", "-3.5"],
                }
            )
            mock_api.get_data.return_value = sample_data

            # Mock filter_by_polygon to return empty dataframe (all filtered out)
            mock_api.filter_by_polygon.return_value = pd.DataFrame()

            response = client.post("/ea/water-quality", json=request_data)

            assert response.status_code == 200
            json_response = response.json()
            assert json_response["total_records"] == 2
            assert json_response["filtered_records"] == 0
            assert json_response["data"] == []

    def test_ea_water_quality_empty_response(self):
        """Test EA water quality request with no data returned."""
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
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock empty dataframe
            mock_api.get_data.return_value = pd.DataFrame()

            response = client.post("/ea/water-quality", json=request_data)

            assert response.status_code == 200
            json_response = response.json()
            assert json_response["total_records"] == 0
            assert json_response["filtered_records"] == 0
            assert json_response["data"] == []
            assert "message" in json_response

    def test_ea_water_quality_invalid_dates(self):
        """Test EA water quality request with invalid date format."""
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
            "determinand": "0076",
            "start_date": "invalid-date",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock ValueError for invalid date
            mock_api.get_data.side_effect = ValueError("Invalid date format")

            response = client.post("/ea/water-quality", json=request_data)

            assert response.status_code == 400
            assert "Invalid request parameters" in response.json()["detail"]

    def test_ea_water_quality_missing_area(self):
        """Test EA water quality request without area parameter."""
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
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock ValueError for missing area
            mock_api.get_data.side_effect = ValueError(
                "'area' parameter must be provided"
            )

            response = client.post("/ea/water-quality", json=request_data)

            assert response.status_code == 400

    def test_ea_water_quality_invalid_polygon(self):
        """Test EA water quality request with invalid polygon (too few points)."""
        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.5, 50.3],
                    [-4.5, 51.2],
                ]  # Only 2 points, need at least 3
            },
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        response = client.post("/ea/water-quality", json=request_data)

        # Should fail validation
        assert response.status_code == 422


class TestEAWaterQualityMultipleEndpoint:
    """Tests for /ea/water-quality/multiple endpoint."""

    def test_ea_water_quality_multiple_success(self):
        """Test successful multiple determinands request."""
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
            "determinands": {"0076": "Temperature", "0077": "Conductivity"},
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
            "verbose": False,
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock get_multiple_determinands to return sample data
            sample_data = pd.DataFrame(
                {
                    "phenomenonTime": [
                        "2024-01-01T10:00:00",
                        "2024-01-02T11:00:00",
                    ],
                    "Temperature": ["15.5", "16.2"],
                    "Conductivity": ["100", "110"],
                    "sample.samplingPoint.latitude": ["50.5", "51.0"],
                    "sample.samplingPoint.longitude": ["-4.0", "-3.5"],
                }
            )
            mock_api.get_multiple_determinands.return_value = sample_data

            # Mock filter_by_polygon
            filtered_data = sample_data.iloc[[0]]
            mock_api.filter_by_polygon.return_value = filtered_data

            response = client.post(
                "/ea/water-quality/multiple", json=request_data
            )

            # Verify get_multiple_determinands was called with correct parameters
            mock_api.get_multiple_determinands.assert_called_once_with(
                determinands={"0076": "Temperature", "0077": "Conductivity"},
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="environment_agency,SWX",
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
            assert "determinands" in json_response
            assert "data" in json_response
            assert json_response["total_records"] == 2
            assert json_response["filtered_records"] == 1
            assert "0076" in json_response["determinands"]
            assert "0077" in json_response["determinands"]

    def test_ea_water_quality_multiple_all_filtered_out(self):
        """Test multiple determinands when all data is filtered out by polygon."""

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
            "determinands": {"0076": "Temperature", "0077": "Conductivity"},
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock get_multiple_determinands to return sample data
            sample_data = pd.DataFrame(
                {
                    "phenomenonTime": [
                        "2024-01-01T10:00:00",
                        "2024-01-02T11:00:00",
                    ],
                    "Temperature": ["15.5", "16.2"],
                    "Conductivity": ["100", "110"],
                    "sample.samplingPoint.latitude": ["50.5", "51.0"],
                    "sample.samplingPoint.longitude": ["-4.0", "-3.5"],
                }
            )
            mock_api.get_multiple_determinands.return_value = sample_data

            # Mock filter_by_polygon to return empty dataframe
            mock_api.filter_by_polygon.return_value = pd.DataFrame()

            response = client.post(
                "/ea/water-quality/multiple", json=request_data
            )

            assert response.status_code == 200
            json_response = response.json()
            assert json_response["total_records"] == 2
            assert json_response["filtered_records"] == 0
            assert json_response["data"] == []

    def test_ea_water_quality_multiple_empty_response(self):
        """Test multiple determinands request with no data."""
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
            "determinands": {"0076": "Temperature", "0077": "Conductivity"},
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock empty dataframe
            mock_api.get_multiple_determinands.return_value = pd.DataFrame()

            response = client.post(
                "/ea/water-quality/multiple", json=request_data
            )

            assert response.status_code == 200
            json_response = response.json()
            assert json_response["total_records"] == 0
            assert json_response["filtered_records"] == 0
            assert json_response["data"] == []
            assert "message" in json_response

    def test_ea_water_quality_multiple_empty_determinands(self):
        """Test multiple determinands with empty determinands dict."""
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
            "determinands": {},  # Empty dict
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock empty result
            mock_api.get_multiple_determinands.return_value = pd.DataFrame()

            response = client.post(
                "/ea/water-quality/multiple", json=request_data
            )

            assert response.status_code == 200


class TestPolygonCoordinateHandling:
    """Tests for polygon coordinate transformation and edge cases."""

    def test_polygon_coordinate_format_conversion(self):
        """Test that polygon coordinates are correctly converted to tuples."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-1.0, 50.0],
                    [-1.0, 51.0],
                    [0.0, 51.0],
                    [0.0, 50.0],
                    [-1.0, 50.0],
                ]
            },
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,EA",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Return non-empty DataFrame so filter_by_polygon is called
            sample_data = pd.DataFrame(
                {
                    "id": ["1"],
                    "result": ["15.5"],
                    "phenomenonTime": ["2024-01-01T10:00:00"],
                    "sample.samplingPoint.latitude": ["50.5"],
                    "sample.samplingPoint.longitude": ["-0.5"],
                }
            )
            mock_api.get_data.return_value = sample_data
            mock_api.filter_by_polygon.return_value = sample_data

            response = client.post("/ea/water-quality", json=request_data)

            # Verify polygon coordinates were converted from lists to tuples
            mock_api.filter_by_polygon.assert_called_once()
            call_args = mock_api.filter_by_polygon.call_args
            polygon_arg = call_args[0][1]

            # Check it's a list of tuples (not list of lists)
            assert all(isinstance(coord, tuple) for coord in polygon_arg)
            assert polygon_arg == [
                (-1.0, 50.0),
                (-1.0, 51.0),
                (0.0, 51.0),
                (0.0, 50.0),
                (-1.0, 50.0),
            ]

            assert response.status_code == 200

    def test_polygon_with_float_coordinates(self):
        """Test handling of polygon with various float coordinate formats."""

        request_data = {
            "polygon": {
                "coordinates": [
                    [-4.567891, 50.123456],
                    [-4.567891, 51.234567],
                    [-3.456789, 51.234567],
                    [-3.456789, 50.123456],
                    [-4.567891, 50.123456],
                ]
            },
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            sample_data = pd.DataFrame({"id": ["1"]})
            mock_api.get_data.return_value = sample_data
            mock_api.filter_by_polygon.return_value = sample_data

            response = client.post("/ea/water-quality", json=request_data)

            # Verify high-precision coordinates are preserved
            call_args = mock_api.filter_by_polygon.call_args
            polygon_arg = call_args[0][1]
            assert polygon_arg[0] == (-4.567891, 50.123456)

            assert response.status_code == 200


class TestAPIRootAndHealth:
    """Tests for root and health check endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        json_response = response.json()
        assert "message" in json_response
        assert "version" in json_response
        assert "endpoints" in json_response
        assert "/ea/water-quality" in json_response["endpoints"]
        assert "/ea/water-quality/multiple" in json_response["endpoints"]

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "healthy"


class TestRequestModels:
    """Tests for request model validation."""

    def test_polygon_request_validation(self):
        """Test polygon request requires at least 3 coordinates."""
        request_data = {
            "polygon": {"coordinates": [[-4.5, 50.3], [-4.5, 51.2]]},
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        response = client.post("/ea/water-quality", json=request_data)
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test request validation for missing required fields."""
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
            "determinand": "0076",
            # Missing start_date, end_date, area
        }

        response = client.post("/ea/water-quality", json=request_data)
        assert response.status_code == 422

    def test_verbose_default_value(self):
        """Test verbose parameter defaults to False."""
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
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
            # verbose not provided
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api
            mock_api.get_data.return_value = pd.DataFrame()

            response = client.post("/ea/water-quality", json=request_data)

            assert response.status_code == 200
            # Check that verbose was passed as False (default)
            call_kwargs = mock_api.get_data.call_args[1]
            assert call_kwargs["verbose"] is False


class TestErrorHandling:
    """Tests for error handling."""

    def test_import_error_handling(self):
        """Test handling of ImportError (e.g., shapely not installed)."""
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
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock ImportError
            mock_api.get_data.return_value = pd.DataFrame(
                {
                    "id": ["1"],
                    "result": ["15.5"],
                    "sample.samplingPoint.latitude": ["50.5"],
                    "sample.samplingPoint.longitude": ["-4.0"],
                }
            )
            mock_api.filter_by_polygon.side_effect = ImportError(
                "shapely is required"
            )

            response = client.post("/ea/water-quality", json=request_data)

            assert response.status_code == 500
            assert "Missing required dependency" in response.json()["detail"]

    def test_generic_exception_handling(self):
        """Test handling of generic exceptions."""
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
            "determinand": "0076",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock generic exception
            mock_api.get_data.side_effect = Exception("Unexpected error")

            response = client.post("/ea/water-quality", json=request_data)

            assert response.status_code == 500
            assert "An error occurred" in response.json()["detail"]

    def test_value_error_handling_explicit(self):
        """Test explicit ValueError handling with bad parameter values."""

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
            "determinand": "INVALID",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "area": "environment_agency,SWX",
        }

        with patch("eoflow.api.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api_class.return_value = mock_api

            # Mock ValueError from API validation
            mock_api.get_data.side_effect = ValueError(
                "Invalid determinand code"
            )

            response = client.post("/ea/water-quality", json=request_data)

            assert response.status_code == 400
            assert "Invalid request parameters" in response.json()["detail"]
            assert "Invalid determinand code" in response.json()["detail"]
