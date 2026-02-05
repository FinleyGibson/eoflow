"""
Example Script: Environment Agency Water Quality API

This script demonstrates the basic usage of the EA Water Quality API client
to fetch water quality monitoring data from the Environment Agency.

Common Determinands:
- 0076: Temperature of Water
- 0077: Conductivity at 25°C
- 0180: Orthophosphate, reactive as P
- 6396: Turbidity (NTU)

Common Area Codes:
- "environment_agency,DCS": Environment Agency DCS area
- "environment_agency,SWX": Southwest region (Devon, Cornwall, Somerset, Dorset)
- "local_authority,E06000002": Example local authority area

API Documentation:
https://environment.data.gov.uk/water-quality-beta/api-docs
"""

from datetime import datetime

from eoflow.ea import EAWaterQualityAPI, get_ea_water_quality


def example_1_simple_query():
    """Example 1: Simple query using convenience function"""
    print("\n=== Example 1: Simple Water Temperature Query ===")

    # Fetch water temperature data for Southwest region
    df = get_ea_water_quality(
        determinand="0076",  # Water Temperature
        start_date="2024-01-01",
        end_date="2024-01-31",
        area="environment_agency,SWX",  # Southwest region
        verbose=True,
    )

    print(f"\nRetrieved {len(df)} observations")
    print(f"\nFirst few records:")
    print(df.head())


def example_2_api_class():
    """Example 2: Using the API class directly"""
    print("\n=== Example 2: Using API Class with Custom Settings ===")

    # Initialize API client with custom settings
    api = EAWaterQualityAPI(
        delay=0.5,  # 0.5 second delay between requests
        timeout=30,  # 30 second timeout
    )

    # Fetch conductivity data
    df = api.get_data(
        determinand="0077",  # Conductivity at 25°C
        start_date="2024-01-01",
        end_date="2024-01-31",
        area="environment_agency,SWX",
        verbose=False,
    )

    print(f"\nRetrieved {len(df)} conductivity observations")
    if not df.empty:
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")


def example_3_multiple_determinands():
    """Example 3: Fetching multiple determinands"""
    print("\n=== Example 3: Multiple Determinands ===")

    api = EAWaterQualityAPI()

    # Define multiple determinands to fetch
    determinands = {
        "0076": "Temperature",
        "0077": "Conductivity",
        "0180": "Orthophosphate",
    }

    df = api.get_multiple_determinands(
        determinands=determinands,
        start_date="2024-01-01",
        end_date="2024-01-31",
        area="environment_agency,SWX",
        verbose=False,
    )

    print(f"\nRetrieved {len(df)} observations")
    if not df.empty:
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few records:")
        print(df[["Date", "Temperature", "Conductivity", "Orthophosphate"]].head())


def example_4_polygon_filtering():
    """Example 4: Filtering results by geographic polygon"""
    print("\n=== Example 4: Polygon Filtering ===")

    api = EAWaterQualityAPI()

    # First, get data for a broad area
    df = api.get_data(
        determinand="0076",
        start_date="2024-01-01",
        end_date="2024-01-31",
        area="environment_agency,SWX",
        verbose=False,
    )

    print(f"\nTotal observations before filtering: {len(df)}")

    # Define a polygon (example: rough box around Devon)
    # Format: list of (longitude, latitude) tuples
    devon_polygon = [
        (-4.5, 50.3),
        (-4.5, 51.2),
        (-3.0, 51.2),
        (-3.0, 50.3),
        (-4.5, 50.3),  # Close the polygon
    ]

    try:
        # Filter observations within the polygon
        filtered_df = api.filter_by_polygon(df, devon_polygon)
        print(f"Observations after polygon filtering: {len(filtered_df)}")

        if not filtered_df.empty:
            print(f"\nSample locations:")
            print(
                filtered_df[
                    [
                        "sample.samplingPoint.notation",
                        "sample.samplingPoint.latitude",
                        "sample.samplingPoint.longitude",
                    ]
                ].head()
            )
    except ImportError:
        print("\nNote: shapely package required for polygon filtering")
        print("Install with: pip install shapely")


def example_5_data_analysis():
    """Example 5: Basic data analysis"""
    print("\n=== Example 5: Basic Data Analysis ===")

    api = EAWaterQualityAPI()

    df = api.get_data(
        determinand="0076",  # Temperature
        start_date="2024-01-01",
        end_date="2024-03-31",
        area="environment_agency,SWX",
        verbose=False,
    )

    if not df.empty:
        # Convert result to numeric
        df["result_numeric"] = pd.to_numeric(df["result"], errors="coerce")

        print(f"\nTemperature Statistics:")
        print(f"  Total observations: {len(df)}")
        print(f"  Mean temperature: {df['result_numeric'].mean():.2f}°C")
        print(f"  Min temperature: {df['result_numeric'].min():.2f}°C")
        print(f"  Max temperature: {df['result_numeric'].max():.2f}°C")
        print(f"  Std deviation: {df['result_numeric'].std():.2f}°C")

        # Count by sampling point
        print(f"\nTop 5 sampling points by number of observations:")
        top_points = df["sample.samplingPoint.notation"].value_counts().head()
        print(top_points)


def main():
    """Run all examples"""
    print("=" * 60)
    print("EA Water Quality API Examples")
    print("=" * 60)

    try:
        # Run examples (comment out any you don't want to run)
        example_1_simple_query()
        example_2_api_class()
        example_3_multiple_determinands()
        example_4_polygon_filtering()
        example_5_data_analysis()

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: These examples require network access to the EA API")
        print("Some queries may return no data depending on availability")


if __name__ == "__main__":
    import pandas as pd

    main()
