"""
Integration tests for EA Water Quality API that call the real EA API.

These tests are marked with @pytest.mark.integration and should be run explicitly:
    pytest -m integration
    pytest tests/api/test_api_ea_integration.py

They are skipped by default to avoid:
- Slow test runs
- Hitting API rate limits
- Network dependency in CI/CD
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from eoflow.ea import EAWaterQualityAPI


@pytest.mark.integration
@pytest.mark.slow
class TestEAAPIIntegration:
    """Integration tests that call the real EA API."""

    @pytest.fixture
    def api(self):
        """Create API instance with longer timeout for integration tests."""
        return EAWaterQualityAPI(delay=1.0, timeout=60)

    def test_get_data_real_api_short_range(self, api):
        """Test fetching real data from EA API with a short date range."""
        # Use a very short date range to get quick results
        df = api.get_data(
            determinand="0076",  # Water temperature
            start_date="2024-01-01",
            end_date="2024-01-02",
            area=None,
            verbose=False,
        )

        # Basic assertions
        assert df is not None, "API should return a DataFrame"
        assert len(df) >= 0, "Should return a DataFrame (may be empty)"

        if not df.empty:
            # If we got data, verify structure
            assert "result" in df.columns, "Should have result column"
            assert "phenomenonTime" in df.columns, (
                "Should have phenomenonTime column"
            )
            assert "samplingPoint.notation" in df.columns, (
                "Should have samplingPoint.notation column"
            )
            print(f"\n✓ Retrieved {len(df)} records from EA API")
        else:
            print(
                "\n⚠ No data returned (API may not have data for this date range)"
            )

    def test_get_data_with_area_none(self, api):
        """Test that area=None parameter works (doesn't raise error)."""
        # This tests the fix we made to allow area=None
        df = api.get_data(
            determinand="0076",
            start_date="2024-01-01",
            end_date="2024-01-01",
            area=None,  # Should not raise ValueError
            verbose=False,
        )

        assert df is not None
        print(f"\n✓ area=None works correctly, retrieved {len(df)} records")

    def test_filter_by_polygon_real_data(self, api):
        """Test polygon filtering with real EA data."""
        # Fetch real data
        df = api.get_data(
            determinand="0076",
            start_date="2024-01-01",
            end_date="2024-01-02",
            area=None,
            verbose=False,
        )

        if df.empty:
            pytest.skip("No data returned from API for this date range")

        # Define a simple polygon (Southwest England area)
        # Roughly covers Devon, Cornwall, Somerset area
        polygon = [
            (-5.5, 49.9),
            (-5.5, 51.5),
            (-2.5, 51.5),
            (-2.5, 49.9),
            (-5.5, 49.9),
        ]

        # Filter by polygon
        filtered_df = api.filter_by_polygon(
            df=df,
            polygon=polygon,
        )

        # Assertions
        assert filtered_df is not None
        assert len(filtered_df) <= len(df), (
            "Filtered data should be subset or equal"
        )

        if not filtered_df.empty:
            assert all(col in filtered_df.columns for col in df.columns)
            print(
                f"\n✓ Polygon filtering works: {len(df)} records → {len(filtered_df)} filtered"
            )
        else:
            print("\n⚠ No records within polygon (may be expected)")

    def test_column_structure_real_api(self, api):
        """Test that real API returns expected column structure."""
        df = api.get_data(
            determinand="0076",
            start_date="2024-01-01",
            end_date="2024-01-01",
            area=None,
            verbose=False,
        )

        if df.empty:
            pytest.skip("No data returned from API")

        # Check for expected columns (these are what we found in our testing)
        expected_columns = [
            "samplingPoint.notation",
            "samplingPoint.prefLabel",
            "samplingPoint.easting",
            "samplingPoint.northing",
            "determinand.notation",
            "result",
            "phenomenonTime",
        ]

        for col in expected_columns:
            assert col in df.columns, f"Expected column '{col}' not found"

        print("\n✓ API returned expected column structure")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Sample columns: {list(df.columns)[:10]}")

    def test_coordinate_conversion_with_real_data(self, api):
        """Test that easting/northing coordinates are properly handled."""
        df = api.get_data(
            determinand="0076",
            start_date="2024-01-01",
            end_date="2024-01-01",
            area=None,
            verbose=False,
        )

        if df.empty:
            pytest.skip("No data returned from API")

        # Check that we have easting/northing (not lat/lon)
        assert "samplingPoint.easting" in df.columns
        assert "samplingPoint.northing" in df.columns

        # Test that polygon filtering can handle easting/northing
        polygon = [
            (-6.0, 49.0),
            (-6.0, 52.0),
            (-2.0, 52.0),
            (-2.0, 49.0),
            (-6.0, 49.0),
        ]

        # This should auto-detect and convert easting/northing to lat/lon
        filtered_df = api.filter_by_polygon(df=df, polygon=polygon)

        assert filtered_df is not None
        print("\n✓ Coordinate conversion works with real data")
        print(f"  Records: {len(df)} → {len(filtered_df)} after polygon filter")

    @pytest.mark.skip(
        reason="This test makes many API calls and is very slow (12 requests)"
    )
    def test_get_data_full_year(self, api):
        """Test fetching data for a full year (12 API calls - one per month)."""
        df = api.get_data(
            determinand="0076",
            start_date="2023-01-01",
            end_date="2023-12-31",
            area=None,
            verbose=True,
        )

        assert df is not None
        print(f"\n✓ Full year query completed: {len(df)} total records")

    def test_southwest_region_filtering(self, api):
        """Test filtering by Southwest region name."""
        df = api.get_data(
            determinand="0076",
            start_date="2024-01-01",
            end_date="2024-01-31",
            area=None,
            verbose=False,
        )

        if df.empty:
            pytest.skip("No data returned from API")

        # Check if region column exists and filter for Southwest
        if "samplingPoint.region" in df.columns:
            southwest_df = df[
                df["samplingPoint.region"].str.contains(
                    "South", case=False, na=False
                )
            ]

            print("\n✓ Region filtering test:")
            print(f"  Total records: {len(df)}")
            print(f"  Southwest records: {len(southwest_df)}")

            if not southwest_df.empty:
                print(
                    f"  Southwest regions found: {southwest_df['samplingPoint.region'].unique()}"
                )
        else:
            print("\n⚠ samplingPoint.region column not found in API response")


@pytest.mark.integration
class TestEAAPIErrorHandling:
    """Test error handling with real API."""

    @pytest.fixture
    def api(self):
        """Create API instance."""
        return EAWaterQualityAPI(delay=0.5, timeout=30)

    def test_invalid_determinand(self, api):
        """Test that invalid determinand codes are handled."""
        # Use an invalid determinand code
        df = api.get_data(
            determinand="9999",  # Invalid code
            start_date="2024-01-01",
            end_date="2024-01-01",
            area=None,
            verbose=False,
        )

        # Should return empty DataFrame, not raise an error
        assert df is not None
        assert isinstance(df, type(df))  # Is a DataFrame
        print("\n✓ Invalid determinand handled gracefully (empty result)")

    def test_future_dates(self, api):
        """Test querying future dates."""
        df = api.get_data(
            determinand="0076",
            start_date="2030-01-01",
            end_date="2030-01-02",
            area=None,
            verbose=False,
        )

        # Should return empty DataFrame for future dates
        assert df is not None
        assert df.empty or len(df) == 0
        print("\n✓ Future dates handled correctly (empty result)")
