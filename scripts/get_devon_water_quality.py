"""
Get water quality data for Southwest England and Devon using shapefile filtering.

This script demonstrates how to:
1. Load the Counties and Unitary Authorities shapefile
2. Extract the Devon boundary geometry
3. Fetch water quality data from the Environment Agency API
4. Filter to Southwest England region only
5. Filter Southwest data to only include samples within Devon boundaries
6. Save two CSV files: one for all Southwest samples, one for Devon only

This approach works around the issue where precanned area codes like
'environment_agency,SWX' return 400 errors from the EA API by:
- Fetching data without area filters
- Filtering by region name to get Southwest samples only
- Using shapefile-based polygon filtering to select Devon samples

Output files:
    - southwest_water_quality_samples.csv: All Southwest England samples
    - devon_water_quality.csv: Devon-only samples (subset of Southwest)

IMPORTANT LIMITATION:
    The EA API has a 2500 record limit per request. When fetching all areas
    (area=None), you may not get Southwest/Devon data in the first 2500 records.

    For reliable Devon data extraction, consider:
    - Using shorter date ranges (e.g., 1 week at a time)
    - Fetching multiple months and combining results
    - Or using the original date range but being aware you may get 0 records

Usage:
    python scripts/get_devon_water_quality.py

Note: You may need to adjust the shapefile_path in the main() function
to match your local data directory.
"""

from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import pandas as pd

from eoflow.ea import EAWaterQualityAPI
from eoflow.utils import load_shapefile
from eoflow.log_utils import get_logger

logger = get_logger(__name__)


def get_polygon_coords(gdf: gpd.GeoDataFrame) -> List[Tuple[float, float]]:
    """
    Extract polygon coordinates from a GeoDataFrame.

    Returns a list of (lon, lat) tuples suitable for the EA API filter.
    If the geometry is a MultiPolygon, uses the largest polygon.
    """
    geom = gdf.geometry.iloc[0]

    # Handle MultiPolygon by taking the largest polygon
    if geom.geom_type == 'MultiPolygon':
        logger.info("Geometry is MultiPolygon, extracting largest polygon")
        # Get the polygon with the largest area
        largest_poly = max(geom.geoms, key=lambda p: p.area)
        coords = list(largest_poly.exterior.coords)
    elif geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
    else:
        raise ValueError(f"Unexpected geometry type: {geom.geom_type}")

    # Return as list of (lon, lat) tuples
    return coords


def main():
    """Main function to fetch and filter Southwest and Devon water quality data.

    Returns:
        tuple: (southwest_data, devon_data) - DataFrames with Southwest samples and Devon-filtered samples
    """

    # Configuration
    # Update this path to match your local data directory
    shapefile_path = Path("/home/finley/Work/RDS/projects/enforce/data/Counties_and_Unitary_Authorities_December_2023_Boundaries")

    # Determinand codes:
    # 0076 = Water Temperature
    # 0077 = Conductivity at 25°C
    # 0180 = Orthophosphate, reactive as P
    # 6396 = Turbidity (NTU)
    determinand = "6396"  # Water turbidity
    # NOTE: Using a shorter date range to increase likelihood of getting Southwest data
    # The API limit is 2500 records, and all-area queries often return Anglian region first
    start_date = "2020-01-01"
    end_date = "2020-06-01"  # 1 week to start

    print(f"\nFetching Devon water quality data...")
    print(f"Determinand: {determinand}")
    print(f"Date range: {start_date} to {end_date}\n")

    # Step 1: Load the shapefile
    logger.info(f"Loading shapefile from {shapefile_path}")
    try:
        gdf = load_shapefile(shapefile_path)
        logger.info(f"Loaded shapefile with {len(gdf)} features")
        logger.info(f"Available regions: {sorted(gdf['CTYUA23NM'].unique())}")
    except FileNotFoundError as e:
        logger.error(f"Shapefile not found: {e}")
        print(f"\nERROR: Could not find shapefile at {shapefile_path}")
        print("Please update the shapefile_path variable in the script.")
        return None, None

    # Step 2: Select Devon
    logger.info("Filtering for Devon")
    devon_gdf = gdf[gdf["CTYUA23NM"] == "Devon"]

    if devon_gdf.empty:
        logger.error("Devon not found in shapefile!")
        return None, None

    logger.info(f"Devon boundary found with CRS: {devon_gdf.crs}")

    # Ensure the GeoDataFrame is in WGS84 (EPSG:4326) for lat/lon coordinates
    if devon_gdf.crs != "EPSG:4326":
        logger.info(f"Reprojecting from {devon_gdf.crs} to EPSG:4326")
        devon_gdf = devon_gdf.to_crs("EPSG:4326")

    # Step 3: Extract polygon coordinates
    polygon_coords = get_polygon_coords(devon_gdf)
    logger.info(f"Extracted polygon with {len(polygon_coords)} vertices")

    # Get bounding box for reference
    bounds = devon_gdf.total_bounds  # minx, miny, maxx, maxy
    logger.info(f"Devon bounding box: lon=[{bounds[0]:.4f}, {bounds[2]:.4f}], "
                f"lat=[{bounds[1]:.4f}, {bounds[3]:.4f}]")

    # Step 4: Fetch water quality data
    logger.info(f"Fetching water quality data for determinand {determinand}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("Note: Fetching all areas (no area filter) to avoid API limitations")
    print("\nFetching data from Environment Agency API...")
    print("This may take several minutes depending on the date range...")
    print("\nIMPORTANT: The API has a 2500 record limit. When fetching all areas,")
    print("you may not get Southwest/Devon data. Use shorter date ranges for better results.\n")

    api = EAWaterQualityAPI(delay=0.5, timeout=30)

    try:
        # Get data without area filter (since precanned areas don't seem to work)
        # This will fetch ALL water quality data for the determinand, which we'll
        # then filter using the Devon polygon
        df = api.get_data(
            determinand=determinand,
            start_date=start_date,
            end_date=end_date,
            area="environment_agency,DCS",
            verbose=True,
        )
    except Exception as e:
        logger.error(f"Failed to fetch data from EA API: {e}")
        print(f"\nERROR: Could not fetch data from EA API: {e}")
        return None, None

    if df.empty:
        logger.warning("No data retrieved from API")
        print("\n✗ No data retrieved. Try a different date range.")
        return None, None

    logger.info(f"Retrieved {len(df)} total observations")

    # Check what regions we got
    if "samplingPoint.region" in df.columns:
        regions = df["samplingPoint.region"].value_counts()
        logger.info(f"Regions in data: {regions.to_dict()}")
        print(f"\n  Regions in data:")
        for region, count in regions.items():
            print(f"    - {region}: {count:,} records")

    # Step 5: Filter to Southwest England only
    logger.info("Filtering to Southwest region")
    print("\nFiltering to Southwest England...")

    if "samplingPoint.region" in df.columns:
        # Filter for Southwest region
        southwest_df = df[df["samplingPoint.region"].str.contains("South", case=False, na=False)]
        logger.info(f"Found {len(southwest_df)} Southwest observations")
        print(f"  Southwest records: {len(southwest_df):,}")

        if southwest_df.empty:
            logger.warning("No Southwest region data in results - try different dates or shorter range")
            print("\n✗ No Southwest data found. Try different dates or shorter date range.")
            return None, None
    else:
        logger.warning("No region column found - cannot filter to Southwest")
        southwest_df = df

    # Save Southwest samples to CSV
    southwest_samples_path = Path("southwest_water_quality_samples.csv")
    southwest_df.to_csv(southwest_samples_path, index=False)
    logger.info(f"Southwest samples saved to {southwest_samples_path}")
    print(f"\n✓ Southwest samples saved to {southwest_samples_path}")
    print(f"  Total Southwest records: {len(southwest_df):,}")

    # Step 6: Filter by Devon polygon
    logger.info("Filtering Southwest observations to Devon boundaries")
    print("\nFiltering Southwest data to Devon boundaries...")
    devon_df = api.filter_by_polygon(
        df=southwest_df,  # Filter from Southwest data, not all data
        polygon=polygon_coords,
        # Columns will be auto-detected (easting/northing or lat/lon)
    )

    if devon_df.empty:
        logger.warning("No observations found within Devon boundaries")
        print("\n✗ No Devon observations found in this data.")
        print("   The API returned data from other regions (2500 record limit).")
        print("   Try:")
        print("   - A shorter date range (e.g., 1 week instead of 1 month)")
        print("   - Different dates (summer months may have more samples)")
        print("   - Multiple smaller queries combined together")
        return southwest_df, None

    logger.info(f"Found {len(devon_df)} observations within Devon")

    # Step 7: Display summary statistics
    print("\n" + "="*80)
    print("DEVON WATER QUALITY DATA SUMMARY")
    print("="*80)
    print(f"Determinand: {determinand} (Water Temperature)")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Total observations: {len(devon_df)}")

    site_col = "samplingPoint.prefLabel" if "samplingPoint.prefLabel" in devon_df.columns else "samplingPoint.notation"
    if site_col in devon_df.columns:
        unique_sites = devon_df[site_col].nunique()
        print(f"Unique sampling sites: {unique_sites}")

    if "result" in devon_df.columns:
        devon_df["result_numeric"] = pd.to_numeric(devon_df["result"], errors="coerce")
        print(f"\nTemperature statistics (°C):")
        print(f"  Mean: {devon_df['result_numeric'].mean():.2f}")
        print(f"  Min: {devon_df['result_numeric'].min():.2f}")
        print(f"  Max: {devon_df['result_numeric'].max():.2f}")
        print(f"  Median: {devon_df['result_numeric'].median():.2f}")

    print("="*80 + "\n")

    # Display first few rows
    print("First 5 observations:")
    display_cols = [
        "phenomenonTime",
        "samplingPoint.prefLabel",
        "result",
        "samplingPoint.easting",
        "samplingPoint.northing",
    ]
    available_cols = [col for col in display_cols if col in devon_df.columns]
    print(devon_df[available_cols].head())

    return southwest_df, devon_df


if __name__ == "__main__":
    try:
        southwest_data, devon_data = main()

        # Save Devon data to CSV
        if devon_data is not None and len(devon_data) > 0:
            output_path = Path("devon_water_quality.csv")
            devon_data.to_csv(output_path, index=False)
            logger.info(f"Devon data saved to {output_path}")
            print(f"\n✓ Devon data saved to {output_path}")
            print(f"✓ Total Devon records: {len(devon_data):,}")
        else:
            print("\n✗ No Devon data to save")

        # Note: Southwest data already saved in main()
        if southwest_data is not None:
            print(f"\nSummary:")
            print(f"  Southwest samples: {len(southwest_data):,} (saved to southwest_water_quality_samples.csv)")
            if devon_data is not None:
                print(f"  Devon samples: {len(devon_data):,} (saved to devon_water_quality.csv)")
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n✗ ERROR: {e}")
        raise
