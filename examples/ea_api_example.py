"""
Example Script: Using the EA Water Quality API Endpoints

This script demonstrates how to make requests to the FastAPI endpoints
for fetching EA water quality data within a polygon.

To run the API server first:
    uvicorn eoflow.api:app --reload

Then run this script in another terminal:
    python examples/ea_api_example.py

Requirements:
    pip install requests
"""

import json
from typing import Dict, List

import requests


# API base URL (adjust if running on different host/port)
API_BASE_URL = "http://localhost:8000"


def example_1_single_determinand():
    """Example 1: Fetch single determinand within a polygon"""
    print("\n" + "=" * 60)
    print("Example 1: Single Determinand (Water Temperature)")
    print("=" * 60)

    # Define the request payload
    payload = {
        "polygon": {
            "coordinates": [
                [-4.5, 50.3],   # Southwest corner
                [-4.5, 51.2],   # Northwest corner
                [-3.0, 51.2],   # Northeast corner
                [-3.0, 50.3],   # Southeast corner
                [-4.5, 50.3]    # Close the polygon
            ]
        },
        "determinand": "0076",  # Water Temperature
        "start_date": "2024-01-01",
        "end_date": "2025-01-31",
        "area": "environment_agency,SWX",  # Southwest region
        "verbose": False
    }

    # Make the request
    response = requests.post(
        f"{API_BASE_URL}/ea/water-quality",
        json=payload
    )

    # Handle the response
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Request successful!")
        print(f"  Total records from EA API: {data['total_records']}")
        print(f"  Records within polygon: {data['filtered_records']}")
        print(f"  Data points returned: {len(data['data'])}")

        if data['data']:
            print(f"\n  First record:")
            first_record = data['data'][0]
            for key, value in list(first_record.items())[:5]:
                print(f"    {key}: {value}")
    else:
        print(f"\n✗ Request failed with status code: {response.status_code}")
        print(f"  Error: {response.json().get('detail', 'Unknown error')}")


def example_2_multiple_determinands():
    """Example 2: Fetch multiple determinands within a polygon"""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Determinands")
    print("=" * 60)

    # Define the request payload with multiple determinands
    payload = {
        "polygon": {
            "coordinates": [
                [-4.5, 50.3],
                [-4.5, 51.2],
                [-3.0, 51.2],
                [-3.0, 50.3],
                [-4.5, 50.3]
            ]
        },
        "determinands": {
            "0076": "Temperature",       # Water Temperature
            "0077": "Conductivity",      # Conductivity at 25°C
            "0180": "Orthophosphate"     # Orthophosphate as P
        },
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "area": "environment_agency,SWX",
        "verbose": False
    }

    # Make the request
    response = requests.post(
        f"{API_BASE_URL}/ea/water-quality/multiple",
        json=payload
    )

    # Handle the response
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Request successful!")
        print(f"  Determinands fetched: {', '.join(data['determinands'])}")
        print(f"  Total records from EA API: {data['total_records']}")
        print(f"  Records within polygon: {data['filtered_records']}")
        print(f"  Data points returned: {len(data['data'])}")

        if data['data']:
            print(f"\n  First record (selected fields):")
            first_record = data['data'][0]
            for key in ['Temperature', 'Conductivity', 'Orthophosphate']:
                if key in first_record:
                    print(f"    {key}: {first_record[key]}")
    else:
        print(f"\n✗ Request failed with status code: {response.status_code}")
        print(f"  Error: {response.json().get('detail', 'Unknown error')}")


def example_3_custom_polygon():
    """Example 3: Using a custom polygon (smaller area)"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Polygon (Smaller Devon Area)")
    print("=" * 60)

    # Define a smaller polygon focused on a specific area
    payload = {
        "polygon": {
            "coordinates": [
                [-3.8, 50.5],
                [-3.8, 50.8],
                [-3.4, 50.8],
                [-3.4, 50.5],
                [-3.8, 50.5]
            ]
        },
        "determinand": "6396",  # Turbidity
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "area": "environment_agency,SWX",
        "verbose": False
    }

    response = requests.post(
        f"{API_BASE_URL}/ea/water-quality",
        json=payload
    )

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Request successful!")
        print(f"  Total records from EA API: {data['total_records']}")
        print(f"  Records within polygon: {data['filtered_records']}")

        if data['total_records'] > 0:
            reduction = 100 * (1 - data['filtered_records'] / data['total_records'])
            print(f"  Polygon filtering reduced data by: {reduction:.1f}%")
    else:
        print(f"\n✗ Request failed with status code: {response.status_code}")


def example_4_error_handling():
    """Example 4: Demonstrate error handling"""
    print("\n" + "=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)

    # Test 1: Invalid polygon (too few points)
    print("\nTest 1: Invalid polygon (too few points)")
    payload = {
        "polygon": {
            "coordinates": [
                [-4.5, 50.3],
                [-4.5, 51.2]
            ]
        },
        "determinand": "0076",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "area": "environment_agency,SWX"
    }

    response = requests.post(
        f"{API_BASE_URL}/ea/water-quality",
        json=payload
    )
    print(f"  Status: {response.status_code} (expected 422 - validation error)")

    # Test 2: Missing required field
    print("\nTest 2: Missing required field (no area)")
    payload = {
        "polygon": {
            "coordinates": [
                [-4.5, 50.3],
                [-4.5, 51.2],
                [-3.0, 51.2],
                [-3.0, 50.3],
                [-4.5, 50.3]
            ]
        },
        "determinand": "0076",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
        # Missing "area" field
    }

    response = requests.post(
        f"{API_BASE_URL}/ea/water-quality",
        json=payload
    )
    print(f"  Status: {response.status_code} (expected 422 - validation error)")


def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass
    return False


def main():
    """Run all examples"""
    print("=" * 60)
    print("EA Water Quality API Examples")
    print("=" * 60)

    # Check if API is running
    print("\nChecking API status...")
    if not check_api_health():
        print("\n✗ ERROR: API is not running!")
        print("\nPlease start the API server first:")
        print("  uvicorn eoflow.api:app --reload")
        print("\nThen run this script again.")
        return

    print("✓ API is running")

    try:
        # Run examples
        example_1_single_determinand()
        example_2_multiple_determinands()
        example_3_custom_polygon()
        example_4_error_handling()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        print("\nNote: Some examples may return no data depending on")
        print("EA API availability and the specified date ranges.")

    except requests.exceptions.RequestException as e:
        print(f"\n✗ Network error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


# Common determinand codes for reference
DETERMINAND_CODES = {
    "0076": "Temperature of Water",
    "0077": "Conductivity at 25°C",
    "0180": "Orthophosphate, reactive as P",
    "0115": "Dissolved oxygen saturation",
    "0556": "Nitrate as N",
    "6396": "Turbidity (NTU)",
    "0117": "pH",
    "0191": "Total oxidised nitrogen as N"
}

# Common area codes for reference
AREA_CODES = {
    "Southwest": "environment_agency,SWX",
    "DCS": "environment_agency,DCS",
    "Thames": "environment_agency,TH",
    "Anglian": "environment_agency,AN",
    "Midlands": "environment_agency,MD",
    "North East": "environment_agency,NE",
    "North West": "environment_agency,NW",
    "Southern": "environment_agency,SN",
    "Yorkshire": "environment_agency,YK"
}


if __name__ == "__main__":
    main()
