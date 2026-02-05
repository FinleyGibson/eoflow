"""
Tests for the EA Water Quality API module.

Tests cover:
- API initialization and configuration
- Date range generation
- API request handling (with mocking)
- Data fetching and processing
- Multiple determinand fetching
- Polygon filtering
- Error handling
"""

import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eoflow.ea import EAWaterQualityAPI, get_ea_water_quality


class TestEAWaterQualityAPIInit:
    """Tests for API initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config values."""
        api = EAWaterQualityAPI()
        assert api.delay == 1.0
        assert api.timeout == 60
        assert api.max_limit == 2500
        assert (
            api.base_url
            == "https://environment.data.gov.uk/water-quality/data/observation"
        )

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        api = EAWaterQualityAPI(
            delay=1.0,
            timeout=60,
            max_limit=1000,
            base_url="https://example.com",
        )
        assert api.delay == 1.0
        assert api.timeout == 60
        assert api.max_limit == 1000
        assert api.base_url == "https://example.com"

    def test_init_creates_session(self):
        """Test that initialization creates a requests session."""
        api = EAWaterQualityAPI()
        assert hasattr(api, "session")
        assert api.session is not None


class TestMonthRangeGeneration:
    """Tests for month range generation."""

    def test_generate_month_ranges_single_month(self):
        """Test generating ranges for a single month."""
        api = EAWaterQualityAPI()
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        ranges = api._generate_month_ranges(start, end)

        assert len(ranges) == 1
        assert ranges[0][0] == start
        assert ranges[0][1] == end

    def test_generate_month_ranges_multiple_months(self):
        """Test generating ranges for multiple months."""
        api = EAWaterQualityAPI()
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 31)
        ranges = api._generate_month_ranges(start, end)

        assert len(ranges) == 3
        assert ranges[0][0] == datetime(2024, 1, 1)
        assert ranges[0][1] == datetime(2024, 1, 31)
        assert ranges[1][0] == datetime(2024, 2, 1)
        assert ranges[1][1] == datetime(2024, 2, 29)  # Leap year
        assert ranges[2][0] == datetime(2024, 3, 1)
        assert ranges[2][1] == datetime(2024, 3, 31)

    def test_generate_month_ranges_partial_month(self):
        """Test generating ranges when end date is mid-month."""
        api = EAWaterQualityAPI()
        start = datetime(2024, 1, 1)
        end = datetime(2024, 2, 15)
        ranges = api._generate_month_ranges(start, end)

        assert len(ranges) == 2
        assert ranges[0][1] == datetime(2024, 1, 31)
        assert ranges[1][1] == datetime(2024, 2, 15)

    def test_generate_month_ranges_year_boundary(self):
        """Test generating ranges across year boundary."""
        api = EAWaterQualityAPI()
        start = datetime(2023, 12, 1)
        end = datetime(2024, 2, 29)
        ranges = api._generate_month_ranges(start, end)

        assert len(ranges) == 3
        assert ranges[0][0].year == 2023
        assert ranges[0][0].month == 12
        assert ranges[2][0].year == 2024
        assert ranges[2][0].month == 2


class TestMakeRequest:
    """Tests for API request handling."""

    def test_make_request_success(self):
        """Test successful API request."""
        api = EAWaterQualityAPI()

        # Mock CSV response
        csv_data = """id,determinand.notation,result,phenomenonTime
1,0076,15.5,2024-01-01T10:00:00
2,0076,16.2,2024-01-01T11:00:00"""

        with patch.object(api.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.text = csv_data
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = api._make_request(
                determinand="0076",
                date_from="2024-01-01",
                date_to="2024-01-31",
                precanned_area="test_area",
            )

            assert result is not None
            assert len(result) == 2
            assert "result" in result.columns
            assert result["result"].iloc[0] == "15.5"

    def test_make_request_empty_response(self):
        """Test handling of empty API response."""
        api = EAWaterQualityAPI()

        with patch.object(api.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.text = ""
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = api._make_request(
                determinand="0076",
                date_from="2024-01-01",
                date_to="2024-01-31",
                precanned_area="test_area",
            )

            assert result is None

    def test_make_request_api_error(self):
        """Test handling of API request errors."""
        api = EAWaterQualityAPI()

        with patch.object(api.session, "post") as mock_post:
            from requests.exceptions import RequestException

            mock_post.side_effect = RequestException("API Error")

            with pytest.warns(UserWarning):
                result = api._make_request(
                    determinand="0076",
                    date_from="2024-01-01",
                    date_to="2024-01-31",
                    precanned_area="test_area",
                )

            assert result is None

    def test_make_request_parameters(self):
        """Test that correct parameters are passed to API."""
        api = EAWaterQualityAPI()

        with patch.object(api.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.text = "id,result\n1,10.5"
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            api._make_request(
                determinand="0076",
                date_from="2024-01-01",
                date_to="2024-01-31",
                precanned_area="test_area",
                limit=1000,
            )

            # Check that post was called with correct parameters
            call_args = mock_post.call_args
            assert call_args[1]["params"]["determinand"] == "0076"
            assert call_args[1]["params"]["dateFrom"] == "2024-01-01"
            assert call_args[1]["params"]["dateTo"] == "2024-01-31"
            assert call_args[1]["params"]["limit"] == 1000
            assert call_args[1]["params"]["precannedArea"] == "test_area"


class TestGetData:
    """Tests for the main get_data method."""

    def test_get_data_without_area(self):
        """Test that area parameter is optional and defaults to fetching all areas."""
        api = EAWaterQualityAPI()

        csv_data = """id,result,phenomenonTime
1,15.5,2024-01-01T10:00:00"""

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = pd.read_csv(
                StringIO(csv_data), dtype=str
            )

            result = api.get_data(
                determinand="0076",
                start_date="2024-01-01",
                end_date="2024-01-31",
                verbose=False,
            )

            # Check that _make_request was called with None for precanned_area
            call_args = mock_request.call_args
            assert call_args[1]["precanned_area"] is None
            assert len(result) == 1

    def test_get_data_with_area(self):
        """Test fetching data with area parameter."""
        api = EAWaterQualityAPI()

        csv_data = """id,determinand.notation,result,phenomenonTime
1,0076,15.5,2024-01-01T10:00:00
2,0076,16.2,2024-01-02T11:00:00"""

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = pd.read_csv(
                StringIO(csv_data), dtype=str
            )

            result = api.get_data(
                determinand="0076",
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="test_area",
                verbose=False,
            )

            assert len(result) == 2
            assert "result" in result.columns
            assert "phenomenonTime" in result.columns
            assert "Date" in result.columns

    def test_get_data_with_custom_area(self):
        """Test fetching data with custom area code."""
        api = EAWaterQualityAPI()

        csv_data = """id,result,phenomenonTime
1,15.5,2024-01-01T10:00:00"""

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = pd.read_csv(
                StringIO(csv_data), dtype=str
            )

            result = api.get_data(
                determinand="0076",
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="environment_agency,SWX",
                verbose=False,
            )

            # Check that area was passed correctly
            call_args = mock_request.call_args
            assert call_args[1]["precanned_area"] == "environment_agency,SWX"
            assert len(result) == 1

    def test_get_data_calls_make_request_correctly(self):
        """Test that get_data calls _make_request with correct parameters."""
        api = EAWaterQualityAPI()

        csv_data = """id,result,phenomenonTime
1,15.5,2024-01-01T10:00:00"""

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = pd.read_csv(
                StringIO(csv_data), dtype=str
            )

            result = api.get_data(
                determinand="0076",
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="test_area",
                verbose=False,
            )

            # Check _make_request was called with precanned_area
            mock_request.assert_called()
            assert len(result) == 1

    def test_get_data_datetime_conversion(self):
        """Test that string dates are converted to datetime."""
        api = EAWaterQualityAPI()

        csv_data = """id,result,phenomenonTime
1,15.5,2024-01-01T10:00:00"""

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = pd.read_csv(
                StringIO(csv_data), dtype=str
            )

            # Test with string dates
            result = api.get_data(
                determinand="0076",
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="test",
                verbose=False,
            )

            assert len(result) == 1

            # Test with datetime objects
            result2 = api.get_data(
                determinand="0076",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
                area="test",
                verbose=False,
            )

            assert len(result2) == 1

    def test_get_data_drops_missing_results(self):
        """Test that rows with missing results are dropped."""
        api = EAWaterQualityAPI()

        csv_data = """id,result,phenomenonTime
1,15.5,2024-01-01T10:00:00
2,,2024-01-02T11:00:00
3,16.2,2024-01-03T12:00:00"""

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = pd.read_csv(StringIO(csv_data))

            result = api.get_data(
                determinand="0076",
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="test",
                verbose=False,
            )

            # Should have 2 rows (row with missing result dropped)
            assert len(result) == 2

    def test_get_data_no_data_returned(self):
        """Test handling when no data is returned from API."""
        api = EAWaterQualityAPI()

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = None

            with pytest.warns(UserWarning):
                result = api.get_data(
                    determinand="0076",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    area="test",
                    verbose=False,
                )

            assert result.empty

    def test_get_data_multiple_months(self):
        """Test that multiple months are requested and combined."""
        api = EAWaterQualityAPI(delay=0)  # No delay for testing

        csv_data = """id,result,phenomenonTime
1,15.5,2024-01-01T10:00:00"""

        with patch.object(api, "_make_request") as mock_request:
            mock_request.return_value = pd.read_csv(
                StringIO(csv_data), dtype=str
            )

            result = api.get_data(
                determinand="0076",
                start_date="2024-01-01",
                end_date="2024-03-31",
                area="test",
                verbose=False,
            )

            # Should be called 3 times (Jan, Feb, Mar)
            assert mock_request.call_count == 3


class TestGetMultipleDeterminands:
    """Tests for fetching multiple determinands."""

    def test_get_multiple_determinands_success(self):
        """Test fetching multiple determinands and joining."""
        api = EAWaterQualityAPI()

        # Mock data for first determinand
        csv1 = """id,determinand.notation,result,phenomenonTime,sample.samplingPoint.notation
1,0076,15.5,2024-01-01T10:00:00,SP001
2,0076,16.2,2024-01-02T11:00:00,SP002"""

        # Mock data for second determinand
        csv2 = """id,determinand.notation,result,phenomenonTime,sample.samplingPoint.notation
3,0077,100,2024-01-01T10:00:00,SP001
4,0077,110,2024-01-02T11:00:00,SP002"""

        with patch.object(api, "get_data") as mock_get_data:
            df1 = pd.read_csv(StringIO(csv1), dtype=str)
            df1["phenomenonTime"] = pd.to_datetime(df1["phenomenonTime"])
            df1["Date"] = df1["phenomenonTime"].dt.date

            df2 = pd.read_csv(StringIO(csv2), dtype=str)
            df2["phenomenonTime"] = pd.to_datetime(df2["phenomenonTime"])
            df2["Date"] = df2["phenomenonTime"].dt.date

            mock_get_data.side_effect = [df1, df2]

            determinands = {"0076": "Temperature", "0077": "Conductivity"}

            result = api.get_multiple_determinands(
                determinands=determinands,
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="test",
                verbose=False,
            )

            # Should have renamed columns
            assert "Temperature" in result.columns
            assert "Conductivity" in result.columns

            # Original result column should not exist
            assert "result" not in result.columns

    def test_get_multiple_determinands_no_data(self):
        """Test handling when no data is returned for any determinand."""
        api = EAWaterQualityAPI()

        with patch.object(api, "get_data") as mock_get_data:
            mock_get_data.return_value = pd.DataFrame()

            determinands = {"0076": "Temperature", "0077": "Conductivity"}

            result = api.get_multiple_determinands(
                determinands=determinands,
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="test",
                verbose=False,
            )

            assert result.empty

    def test_get_multiple_determinands_single_determinand(self):
        """Test with only one determinand."""
        api = EAWaterQualityAPI()

        csv = """id,result,phenomenonTime
1,15.5,2024-01-01T10:00:00"""

        with patch.object(api, "get_data") as mock_get_data:
            df = pd.read_csv(StringIO(csv), dtype=str)
            df["phenomenonTime"] = pd.to_datetime(df["phenomenonTime"])
            mock_get_data.return_value = df

            determinands = {"0076": "Temperature"}

            result = api.get_multiple_determinands(
                determinands=determinands,
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="test",
                verbose=False,
            )

            assert "Temperature" in result.columns
            assert len(result) == 1


class TestFilterByPolygon:
    """Tests for polygon filtering."""

    def test_filter_by_polygon_with_list(self):
        """Test filtering with polygon as list of tuples."""
        api = EAWaterQualityAPI()

        # Create test data
        data = {
            "id": [1, 2, 3],
            "sample.samplingPoint.latitude": ["51.5", "52.0", "50.0"],
            "sample.samplingPoint.longitude": ["-1.5", "-1.5", "-3.0"],
            "result": ["10", "20", "30"],
        }
        df = pd.DataFrame(data)

        # Square polygon that should include points 1 and 2 but not 3
        polygon = [
            (-2.0, 51.0),
            (-2.0, 52.5),
            (-1.0, 52.5),
            (-1.0, 51.0),
            (-2.0, 51.0),
        ]

        try:
            result = api.filter_by_polygon(df, polygon)

            # Should keep 2 points
            assert len(result) == 2
            assert 1 in result["id"].values
            assert 2 in result["id"].values
            assert 3 not in result["id"].values
        except ImportError:
            pytest.skip("shapely not installed")

    def test_filter_by_polygon_empty_dataframe(self):
        """Test filtering with empty DataFrame."""
        api = EAWaterQualityAPI()

        df = pd.DataFrame()

        polygon = [
            (-2.0, 51.0),
            (-2.0, 52.5),
            (-1.0, 52.5),
            (-1.0, 51.0),
            (-2.0, 51.0),
        ]

        try:
            result = api.filter_by_polygon(df, polygon)
            assert result.empty
        except ImportError:
            pytest.skip("shapely not installed")

    def test_filter_by_polygon_no_shapely(self):
        """Test that ImportError is raised when shapely not available."""
        api = EAWaterQualityAPI()

        df = pd.DataFrame({
            "id": [1],
            "sample.samplingPoint.latitude": ["51.5"],
            "sample.samplingPoint.longitude": ["-1.5"],
        })
        polygon = [(-2.0, 51.0), (-2.0, 52.0), (-1.0, 51.0), (-2.0, 51.0)]

        # Mock the import to fail
        import sys

        # Save original modules
        shapely_geom = sys.modules.get("shapely.geometry")
        shapely_mod = sys.modules.get("shapely")

        # Remove from sys.modules to force re-import
        if "shapely.geometry" in sys.modules:
            del sys.modules["shapely.geometry"]
        if "shapely" in sys.modules:
            del sys.modules["shapely"]

        # Patch the import to raise ImportError
        with patch.dict(
            "sys.modules", {"shapely.geometry": None, "shapely": None}
        ):
            with pytest.raises(ImportError, match="shapely is required"):
                api.filter_by_polygon(df, polygon)

        # Restore original modules
        if shapely_geom is not None:
            sys.modules["shapely.geometry"] = shapely_geom
        if shapely_mod is not None:
            sys.modules["shapely"] = shapely_mod


class TestConvenienceFunction:
    """Tests for the convenience function."""

    def test_get_ea_water_quality_function(self):
        """Test the convenience function."""
        csv_data = """id,result,phenomenonTime
1,15.5,2024-01-01T10:00:00"""

        with patch("eoflow.ea.EAWaterQualityAPI") as mock_api_class:
            mock_api = Mock()
            mock_api.get_data.return_value = pd.read_csv(
                StringIO(csv_data), dtype=str
            )
            mock_api_class.return_value = mock_api

            result = get_ea_water_quality(
                determinand="0076",
                start_date="2024-01-01",
                end_date="2024-01-31",
                area="test",
                verbose=False,
            )

            # Check that API was instantiated and get_data called
            mock_api_class.assert_called_once()
            mock_api.get_data.assert_called_once()

            # Check parameters passed correctly
            call_kwargs = mock_api.get_data.call_args[1]
            assert call_kwargs["determinand"] == "0076"
            assert call_kwargs["start_date"] == "2024-01-01"
            assert call_kwargs["end_date"] == "2024-01-31"
            assert call_kwargs["area"] == "test"


class TestIntegration:
    """Integration tests (can be slow, marked for optional execution)."""

    @pytest.mark.slow
    def test_real_api_call(self):
        """Test a real API call (requires network and API availability)."""
        pytest.skip("Skipping real API test by default")

        api = EAWaterQualityAPI()

        # Try a small query
        result = api.get_data(
            determinand="0076",
            start_date="2024-01-01",
            end_date="2024-01-01",  # Just one day
            area="environment_agency,DCS",
            verbose=False,
        )

        # Just check that it returned a DataFrame
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
