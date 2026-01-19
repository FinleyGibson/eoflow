"""
EA Water Quality API - Quick Reference Guide

This file provides minimal, copy-paste ready examples for common tasks.
For detailed examples, see ea_water_quality_example.py
"""

from eoflow.ea import EAWaterQualityAPI, get_ea_water_quality


# ============================================================================
# QUICK START - Convenience Function
# ============================================================================

# Fetch single determinand (simplest approach)
df = get_ea_water_quality(
    determinand="0076",              # Water Temperature
    start_date="2024-01-01",
    end_date="2024-01-31",
    area="environment_agency,SWX"    # Southwest region
)


# ============================================================================
# API CLASS - More Control
# ============================================================================

# Initialize with custom settings
api = EAWaterQualityAPI(
    delay=0.5,      # Delay between requests (seconds)
    timeout=30      # Request timeout (seconds)
)

# Fetch single determinand
df = api.get_data(
    determinand="0076",
    start_date="2024-01-01",
    end_date="2024-12-31",
    area="environment_agency,SWX",
    verbose=True
)


# ============================================================================
# MULTIPLE DETERMINANDS
# ============================================================================

api = EAWaterQualityAPI()

determinands = {
    "0076": "Temperature",           # Water temperature
    "0077": "Conductivity",          # Conductivity at 25°C
    "0180": "Orthophosphate",        # Orthophosphate as P
    "6396": "Turbidity"              # Turbidity (NTU)
}

df = api.get_multiple_determinands(
    determinands=determinands,
    start_date="2024-01-01",
    end_date="2024-12-31",
    area="environment_agency,SWX"
)


# ============================================================================
# POLYGON FILTERING (requires shapely)
# ============================================================================

# Get data for broad area
df = api.get_data(
    determinand="0076",
    start_date="2024-01-01",
    end_date="2024-01-31",
    area="environment_agency,SWX"
)

# Define polygon (lon, lat pairs)
polygon = [
    (-4.5, 50.3),    # Southwest corner
    (-4.5, 51.2),    # Northwest corner
    (-3.0, 51.2),    # Northeast corner
    (-3.0, 50.3),    # Southeast corner
    (-4.5, 50.3)     # Close polygon
]

# Filter to polygon
filtered_df = api.filter_by_polygon(df, polygon)


# ============================================================================
# COMMON DETERMINANDS
# ============================================================================

DETERMINANDS = {
    "0076": "Temperature of Water",
    "0077": "Conductivity at 25°C",
    "0180": "Orthophosphate, reactive as P",
    "0115": "Dissolved oxygen saturation",
    "0556": "Nitrate as N",
    "6396": "Turbidity (NTU)",
    "0117": "pH",
    "0191": "Total oxidised nitrogen as N"
}


# ============================================================================
# COMMON AREA CODES
# ============================================================================

AREAS = {
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


# ============================================================================
# WORKING WITH RESULTS
# ============================================================================

import pandas as pd

# Convert result column to numeric
df["result_numeric"] = pd.to_numeric(df["result"], errors="coerce")

# Basic statistics
print(df["result_numeric"].describe())

# Group by sampling point
grouped = df.groupby("sample.samplingPoint.notation")["result_numeric"].agg([
    "count", "mean", "min", "max"
])

# Filter by date
df["Date"] = pd.to_datetime(df["phenomenonTime"]).dt.date
recent = df[df["Date"] >= "2024-06-01"]

# Export to CSV
df.to_csv("water_quality_data.csv", index=False)


# ============================================================================
# ERROR HANDLING
# ============================================================================

try:
    df = get_ea_water_quality(
        determinand="0076",
        start_date="2024-01-01",
        end_date="2024-01-31",
        area="environment_agency,SWX"
    )

    if df.empty:
        print("No data returned for specified parameters")
    else:
        print(f"Retrieved {len(df)} observations")

except ValueError as e:
    print(f"Invalid parameters: {e}")
except Exception as e:
    print(f"Error fetching data: {e}")


# ============================================================================
# TIPS
# ============================================================================

# 1. API automatically handles pagination and monthly chunking
# 2. Large date ranges are split into months automatically
# 3. Missing results are dropped automatically
# 4. Use verbose=False to suppress progress messages
# 5. API respects rate limits with configurable delay between requests
# 6. Results include phenomenonTime (datetime) and Date columns
# 7. For polygon filtering, install: pip install shapely
