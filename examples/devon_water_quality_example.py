"""
Example: Get Water Quality Data for Devon Using Shapefile Filtering

This standalone example demonstrates how to:
1. Load a shapefile containing UK county boundaries
2. Extract the Devon county polygon
3. Fetch water quality data from the Environment Agency API
4. Filter the data to only include samples within Devon

This approach avoids the 400 error from using precanned area codes
by fetching broader data and filtering it using polygon geometry.

Author: Environment Agency Data Team
Date: 2025
"""

import sys
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import pandas as pd

# Add src to path so we can import eoflow modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eoflow.ea import EAWaterQualityAPI
from eoflow.utils import load_shapefile
from eoflow.log_utils import get_logger

logger = get_logger(__name__)


def extract_polygon_from_geodataframe(gdf: gpd.GeoDataFrame) -> List[Tuple[float, float]]:
    """
    Extract polygon coordinates from a GeoDataFrame.

    Args:
        gdf: GeoDataFrame containing a single feature

    Returns:
        List of (lon, lat) coordinate tuples

    Note:
        If the geometry is a MultiPolygon, this extracts the largest polygon.
    """
    geom = gdf.geometry.iloc[0]

    if geom.geom_type == 'MultiPolygon':
        logger.info("Geometry is MultiPolygon, using largest polygon")
        # Extract the polygon with the largest area
        largest_poly = max(geom.geoms, key=lambda p: p.area)
        coords = list(largest_poly.exterior.coords)
    elif geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
    else:
        raise ValueError(f"Unexpected geometry type: {geom.geom_type}")

    return coords


def main():
    """
    Main function to demonstrate fetching and filtering Devon water quality data.
    """

    # ============================================================================
    # CONFIGURATION - Adjust these parameters as needed
    # ============================================================================

    # Path to the Counties and Unitary Authorities shapefile
    # Update this to match your local data directory
    SHAPEFILE_PATH = Path(
        "/home/finley/Work/RDS/projects/enforce/data/"
        "Counties_and_Unitary_Authorities_December_2023_Boundaries"
    )

    # County name to filter (must match the CTYUA23NM field in the shapefile)
    COUNTY_NAME = "Devon"

    # Water quality parameters
    DETERMINAND = "0076"  # Water Temperature (see list below for other codes)
    START_DATE = "2020-01-01"
    END_DATE = "2024-01-31"

    # Common determinand codes:
    # 0076 = Temperature of Water (°C)
    # 0077 = Conductivity at 25°C (µS/cm)
    # 0180 = Orthophosphate, reactive as P (mg/l)
    # 6396 = Turbidity (NTU)
    # 0117 = Dissolved oxygen as O2 (mg/l)
    # 0191 = Total oxidised nitrogen as N (mg/l)

    # API settings
    API_DELAY = 0.5  # Delay between API requests in seconds
    API_TIMEOUT = 30  # Timeout for each request in seconds

    # ============================================================================
    # STEP 1: Load and prepare the shapefile
    # ============================================================================

    print("\n" + "="*80)
    print("DEVON WATER QUALITY DATA EXTRACTION")
    print("="*80 + "\n")

    print(f"Step 1: Loading shapefile...")
    print(f"  Path: {SHAPEFILE_PATH}")

    try:
        # Load the shapefile
        gdf = load_shapefile(SHAPEFILE_PATH)
        print(f"  ✓ Loaded {len(gdf)} features")

        # Check if the county name column exists
        if "CTYUA23NM" not in gdf.columns:
            print(f"\n✗ ERROR: Column 'CTYUA23NM' not found in shapefile")
            print(f"  Available columns: {list(gdf.columns)}")
            return None

    except FileNotFoundError as e:
        print(f"\n✗ ERROR: Shapefile not found")
        print(f"  {e}")
        print(f"\n  Please update SHAPEFILE_PATH in the script.")
        return None

    # ============================================================================
    # STEP 2: Extract the Devon boundary
    # ============================================================================

    print(f"\nStep 2: Extracting {COUNTY_NAME} boundary...")

    # Filter for Devon
    county_gdf = gdf[gdf["CTYUA23NM"] == COUNTY_NAME]

    if county_gdf.empty:
        print(f"\n✗ ERROR: '{COUNTY_NAME}' not found in shapefile")
        print(f"\n  Available regions:")
        for name in sorted(gdf["CTYUA23NM"].unique()):
            print(f"    - {name}")
        return None

    print(f"  ✓ Found {COUNTY_NAME}")
    print(f"  Original CRS: {county_gdf.crs}")

    # Reproject to WGS84 (EPSG:4326) for lat/lon coordinates
    if county_gdf.crs != "EPSG:4326":
        print(f"  Reprojecting to EPSG:4326 (WGS84)...")
        county_gdf = county_gdf.to_crs("EPSG:4326")

    # Extract polygon coordinates
    polygon_coords = extract_polygon_from_geodataframe(county_gdf)
    print(f"  ✓ Polygon extracted: {len(polygon_coords)} vertices")

    # Show bounding box
    bounds = county_gdf.total_bounds  # minx, miny, maxx, maxy
    print(f"  Bounding box:")
    print(f"    Longitude: [{bounds[0]:.4f}, {bounds[2]:.4f}]")
    print(f"    Latitude:  [{bounds[1]:.4f}, {bounds[3]:.4f}]")

    # ============================================================================
    # STEP 3: Fetch water quality data from the Environment Agency API
    # ============================================================================

    print(f"\nStep 3: Fetching water quality data from EA API...")
    print(f"  Determinand: {DETERMINAND}")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"\n  Note: Fetching ALL areas (no geographic filter)")
    print(f"        Data will be filtered by polygon in the next step")
    print(f"\n  This may take several minutes...\n")

    # Initialize the API client
    api = EAWaterQualityAPI(delay=API_DELAY, timeout=API_TIMEOUT)

    try:
        # Fetch data without area filter
        # We don't use area="environment_agency,SWX" because it returns 400 errors
        df = api.get_data(
            determinand=DETERMINAND,
            start_date=START_DATE,
            end_date=END_DATE,
            area=None,  # Fetch all areas
            verbose=True,
        )
    except Exception as e:
        print(f"\n✗ ERROR: Failed to fetch data from EA API")
        print(f"  {e}")
        return None

    if df.empty:
        print(f"\n✗ WARNING: No data retrieved from API")
        print(f"  This could mean:")
        print(f"    - No data available for this determinand and date range")
        print(f"    - API connectivity issues")
        print(f"    - Invalid determinand code")
        return None

    print(f"\n  ✓ Retrieved {len(df):,} total observations")

    # ============================================================================
    # STEP 4: Filter observations to Devon boundaries
    # ============================================================================

    print(f"\nStep 4: Filtering observations to {COUNTY_NAME} boundaries...")

    try:
        county_df = api.filter_by_polygon(
            df=df,
            polygon=polygon_coords,
            # Columns will be auto-detected (easting/northing or lat/lon)
        )
    except Exception as e:
        print(f"\n✗ ERROR: Failed to filter by polygon")
        print(f"  {e}")
        return None

    if county_df.empty:
        print(f"\n✗ WARNING: No observations found within {COUNTY_NAME} boundaries")
        print(f"  This could mean:")
        print(f"    - No sampling points in {COUNTY_NAME} for this determinand")
        print(f"    - Date range doesn't contain data for this region")
        return None

    print(f"  ✓ Found {len(county_df):,} observations within {COUNTY_NAME}")

    # ============================================================================
    # STEP 5: Display summary statistics
    # ============================================================================

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Region: {COUNTY_NAME}")
    print(f"Determinand: {DETERMINAND}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Total observations: {len(county_df):,}")

    # Count unique sampling sites
    site_col = "samplingPoint.prefLabel" if "samplingPoint.prefLabel" in county_df.columns else "samplingPoint.notation"
    if site_col in county_df.columns:
        unique_sites = county_df[site_col].nunique()
        print(f"Unique sampling sites: {unique_sites}")

    # Show value statistics if result column exists
    if "result" in county_df.columns:
        county_df["result_numeric"] = pd.to_numeric(county_df["result"], errors="coerce")
        valid_results = county_df["result_numeric"].dropna()

        if len(valid_results) > 0:
            print(f"\nValue statistics:")
            print(f"  Mean:   {valid_results.mean():.2f}")
            print(f"  Median: {valid_results.median():.2f}")
            print(f"  Min:    {valid_results.min():.2f}")
            print(f"  Max:    {valid_results.max():.2f}")
            print(f"  Std:    {valid_results.std():.2f}")

    print("="*80 + "\n")

    # ============================================================================
    # STEP 6: Display sample data
    # ============================================================================

    print("Sample data (first 10 observations):\n")

    # Select columns to display
    display_cols = [
        "phenomenonTime",
        "samplingPoint.prefLabel",
        "result",
        "samplingPoint.easting",
        "samplingPoint.northing",
    ]

    # Only show columns that exist
    available_cols = [col for col in display_cols if col in county_df.columns]

    # Display the data
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(county_df[available_cols].head(10).to_string(index=False))
    print()

    # ============================================================================
    # STEP 7: Save to CSV
    # ============================================================================

    output_filename = f"{COUNTY_NAME.lower()}_water_quality_{DETERMINAND}.csv"
    output_path = Path(output_filename)

    print(f"Saving data to {output_path}...")
    county_df.to_csv(output_path, index=False)
    print(f"  ✓ Data saved successfully")
    print(f"\n" + "="*80 + "\n")

    return county_df


if __name__ == "__main__":
    try:
        result = main()

        if result is not None:
            print("✓ Script completed successfully!")
        else:
            print("✗ Script completed with warnings - see messages above")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n✗ Script interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        logger.error(f"Unexpected error in main script", exc_info=True)
        sys.exit(1)
