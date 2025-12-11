"""
Environment Agency Water Quality Data API Client

A Python module to fetch water quality monitoring data from the Environment Agency's
Water Quality API. This module handles API pagination, supports multiple determinands,
and can filter data by geographic area (predefined areas or custom polygons).

API Documentation:
- https://gist.github.com/canwaf/2afa25fc6160efb25ac72b7acd60278d
- https://environment.data.gov.uk/water-quality-beta/api-docs

Common Determinands:
- 0076: Temperature of Water
- 0077: Conductivity at 25°C
- 0180: Orthophosphate, reactive as P
- 6396: Turbidity (NTU)

Author: Adapted from R code by Francis Rowney
Last Updated: 2025
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from eoflow.config import get_config
from eoflow.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


class EAWaterQualityAPI:
    """
    Client for fetching water quality data from the Environment Agency API.
    """

    def __init__(
        self,
        delay: Optional[float] = None,
        timeout: Optional[int] = None,
        base_url: Optional[str] = None,
        max_limit: Optional[int] = None,
    ):
        """
        Initialize the API client.

        Args:
            delay: Time to wait between API requests (seconds). If None, uses config value.
            timeout: Request timeout in seconds. If None, uses config value.
            base_url: API base URL. If None, uses config value.
            max_limit: Maximum records per request. If None, uses config value.
        """
        config = get_config()

        self.delay = (
            delay if delay is not None else config.get("api.ea_api_delay")
        )
        self.timeout = (
            timeout if timeout is not None else config.get("api.ea_api_timeout")
        )
        self.base_url = (
            base_url if base_url is not None else config.get("api.ea_base_url")
        )
        self.max_limit = (
            max_limit
            if max_limit is not None
            else config.get("api.ea_max_limit")
        )

        self.session = requests.Session()

        logger.debug(
            f"Initialized EAWaterQualityAPI with delay={self.delay}s, "
            f"timeout={self.timeout}s, max_limit={self.max_limit}"
        )

    def _generate_month_ranges(
        self, start_date: datetime, end_date: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generate a list of month start and end date pairs.

        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval

        Returns:
            List of (month_start, month_end) tuples
        """
        ranges = []
        current = start_date

        while current <= end_date:
            month_end = min(
                current + relativedelta(months=1) - timedelta(days=1), end_date
            )
            ranges.append((current, month_end))
            current = current + relativedelta(months=1)

        return ranges

    def _make_request(
        self,
        determinand: str,
        date_from: str,
        date_to: str,
        area_param: Dict[str, str],
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Make a single API request and return the data as a DataFrame.

        Args:
            determinand: Determinand code (e.g., "0076")
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            area_param: Dictionary with area parameters (e.g., {"precannedArea": "..."} or {"polygon": "..."})
            limit: Maximum number of records to return

        Returns:
            DataFrame with the API response or None if request fails
        """
        if limit is None:
            limit = self.max_limit

        params = {
            "determinand": determinand,
            "dateFrom": date_from,
            "dateTo": date_to,
            "limit": limit,
            **area_param,
        }

        headers = {"Accept": "text/csv"}

        try:
            logger.debug(
                f"Making API request for {date_from} to {date_to}, determinand={determinand}"
            )
            response = self.session.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            # Parse CSV response
            if response.text.strip():
                df = pd.read_csv(StringIO(response.text), dtype=str)
                logger.debug(f"Retrieved {len(df)} records")
                return df
            else:
                logger.warning(f"Empty response for {date_from} to {date_to}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(
                f"Request failed for {date_from} to {date_to}: {str(e)}"
            )
            warnings.warn(
                f"Request failed for {date_from} to {date_to}: {str(e)}"
            )
            return None

    def get_data(
        self,
        determinand: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        area: Optional[str] = None,
        polygon: Optional[Union[Dict, str]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch water quality data for a specific determinand and date range.

        The API has limitations:
        - Maximum 2500 records per call
        - Maximum 1 year of data per request
        - Only one determinand per request

        This method handles these limitations by making multiple requests month-by-month.

        Args:
            determinand: Determinand code (e.g., "0076" for water temperature)
            start_date: Start date (YYYY-MM-DD string or datetime object)
            end_date: End date (YYYY-MM-DD string or datetime object)
            area: Precanned area code. Examples:
                  - "environment_agency,DCS" for EA/Natural England areas
                  - "local_authority,E06000002" for local authority areas
            polygon: GeoJSON polygon (dict or JSON string) defining the area of interest.
                     If provided, takes precedence over 'area' parameter.
            verbose: If True, print progress messages

        Returns:
            DataFrame with water quality observations

        Raises:
            ValueError: If neither area nor polygon is provided
        """
        # Convert string dates to datetime
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Validate area parameters
        if area is None and polygon is None:
            logger.error("Neither area nor polygon provided")
            raise ValueError("Either 'area' or 'polygon' must be provided")

        logger.info(
            f"Fetching data for determinand {determinand} from {start_date.strftime('%Y-%m-%d')} "
            f"to {end_date.strftime('%Y-%m-%d')}"
        )

        # Prepare area parameter
        if polygon is not None:
            if isinstance(polygon, dict):
                polygon_str = json.dumps(polygon)
            else:
                polygon_str = polygon
            area_param = {"polygon": polygon_str}
        else:
            area_param = {"precannedArea": area}

        # Generate month ranges
        month_ranges = self._generate_month_ranges(start_date, end_date)

        # Fetch data for each month
        data_list = []

        for i, (month_start, month_end) in enumerate(month_ranges):
            month_str = month_start.strftime("%Y-%m")
            if verbose:
                logger.info(f"Getting data for: {month_str}")

            df = self._make_request(
                determinand=determinand,
                date_from=month_start.strftime("%Y-%m-%d"),
                date_to=month_end.strftime("%Y-%m-%d"),
                area_param=area_param,
            )

            if df is not None and not df.empty:
                data_list.append(df)

            # Add delay between requests (except for the last one)
            if i < len(month_ranges) - 1:
                time.sleep(self.delay)

        # Combine all months
        if not data_list:
            logger.warning("No data retrieved for the specified parameters")
            warnings.warn("No data retrieved for the specified parameters")
            return pd.DataFrame()

        combined_df = pd.concat(data_list, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} total records")

        # Drop rows with missing results
        combined_df = combined_df.dropna(subset=["result"])

        # Convert phenomenonTime to datetime
        if "phenomenonTime" in combined_df.columns:
            combined_df["phenomenonTime"] = pd.to_datetime(
                combined_df["phenomenonTime"], errors="coerce"
            )
            # Add a separate Date column
            combined_df["Date"] = combined_df["phenomenonTime"].dt.date

        logger.info(f"Successfully retrieved {len(combined_df)} observations")
        return combined_df

    def get_multiple_determinands(
        self,
        determinands: Dict[str, str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        area: Optional[str] = None,
        polygon: Optional[Union[Dict, str]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch data for multiple determinands and join them into a single DataFrame.

        Args:
            determinands: Dictionary mapping determinand codes to column names
                         e.g., {"0076": "Temp Water", "0077": "Cond @ 25C"}
            start_date: Start date (YYYY-MM-DD string or datetime object)
            end_date: End date (YYYY-MM-DD string or datetime object)
            area: Precanned area code
            polygon: GeoJSON polygon defining the area of interest
            verbose: If True, print progress messages

        Returns:
            DataFrame with all determinands joined together
        """
        df_list = []

        for det_code, column_name in determinands.items():
            if verbose:
                logger.info(
                    f"=== Fetching determinand {det_code}: {column_name} ==="
                )

            df = self.get_data(
                determinand=det_code,
                start_date=start_date,
                end_date=end_date,
                area=area,
                polygon=polygon,
                verbose=verbose,
            )

            if not df.empty:
                # Rename result column to the descriptive name
                df = df.rename(columns={"result": column_name})

                # Keep only essential columns for joining
                # Remove determinand-specific columns
                cols_to_drop = [
                    "id",
                    "determinand.notation",
                    "determinand.prefLabel",
                    "unit",
                ]
                cols_to_drop = [
                    col for col in cols_to_drop if col in df.columns
                ]
                df = df.drop(columns=cols_to_drop)

                df_list.append(df)

        # Join all dataframes
        if not df_list:
            logger.warning("No data retrieved for any determinands")
            return pd.DataFrame()

        if len(df_list) == 1:
            return df_list[0]

        # Merge on common columns (all except the renamed result columns)
        logger.info(f"Merging {len(df_list)} determinand datasets")
        result = df_list[0]
        for df in df_list[1:]:
            # Find common columns for merging
            common_cols = list(set(result.columns) & set(df.columns))
            if common_cols:
                result = pd.merge(result, df, on=common_cols, how="outer")

        logger.info(f"Merged dataset shape: {result.shape}")
        return result

    def filter_by_polygon(
        self,
        df: pd.DataFrame,
        polygon: Union[Dict, List[Tuple[float, float]]],
        lat_col: str = "sample.samplingPoint.latitude",
        lon_col: str = "sample.samplingPoint.longitude",
    ) -> pd.DataFrame:
        """
        Filter a DataFrame to only include points within a polygon.

        This is useful when the API doesn't support polygon filtering directly,
        or as a post-processing step to refine results.

        Args:
            df: DataFrame with latitude and longitude columns
            polygon: Either a GeoJSON-style polygon dict or a list of (lon, lat) tuples
            lat_col: Name of the latitude column
            lon_col: Name of the longitude column

        Returns:
            Filtered DataFrame

        Note:
            Requires shapely to be installed: pip install shapely
        """
        try:
            from shapely.geometry import Point, Polygon
        except ImportError:
            logger.error("shapely package not found")
            raise ImportError(
                "shapely is required for polygon filtering. "
                "Install it with: pip install shapely"
            )

        logger.info("Filtering data by polygon")

        # Convert polygon to shapely Polygon
        if isinstance(polygon, dict):
            # Assume GeoJSON format
            if "coordinates" in polygon:
                coords = polygon["coordinates"][0]
            else:
                coords = polygon
        else:
            coords = polygon

        poly = Polygon(coords)

        # Convert lat/lon to numeric
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

        # Filter points within polygon
        mask = df.apply(
            lambda row: poly.contains(Point(row[lon_col], row[lat_col])), axis=1
        )

        filtered_df = df[mask].copy()
        logger.info(
            f"Filtered from {len(df)} to {len(filtered_df)} points within polygon"
        )
        return filtered_df


# Convenience function for simple use cases
def get_ea_water_quality(
    determinand: str,
    start_date: str,
    end_date: str,
    area: Optional[str] = None,
    polygon: Optional[Union[Dict, str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to fetch EA water quality data with minimal setup.

    Uses configuration values from the config file for API settings.

    Args:
        determinand: Determinand code (e.g., "0076")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        area: Precanned area code (e.g., "environment_agency,DCS")
        polygon: GeoJSON polygon for spatial filtering
        verbose: Print progress messages

    Returns:
        DataFrame with water quality data

    Example:
        >>> # Get temperature data for a specific EA area
        >>> df = get_ea_water_quality(
        ...     determinand="0076",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ...     area="environment_agency,DCS"
        ... )
    """
    logger.debug(f"Convenience function called for determinand {determinand}")
    api = EAWaterQualityAPI()
    return api.get_data(
        determinand=determinand,
        start_date=start_date,
        end_date=end_date,
        area=area,
        polygon=polygon,
        verbose=verbose,
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Fetch data for a single determinand using precanned area
    api = EAWaterQualityAPI()

    # Environment Agency area
    temp_data = api.get_data(
        determinand="0076",  # Temperature of Water
        start_date="2024-01-01",
        end_date="2024-03-31",
        area="environment_agency,DCS",
    )

    logger.info(f"Retrieved {len(temp_data)} temperature observations")
    logger.info(f"Sample data:\n{temp_data.head()}")

    # Example 2: Fetch multiple determinands
    determinands = {
        "0076": "Temp Water",
        "0077": "Cond @ 25C",
        "0180": "Orthophospht",
        "6396": "TurbidityNTU",
    }

    combined_data = api.get_multiple_determinands(
        determinands=determinands,
        start_date="2024-01-01",
        end_date="2024-03-31",
        area="local_authority,E06000002",
    )

    logger.info(f"Combined data shape: {combined_data.shape}")

    # Example 3: Using a custom polygon (GeoJSON format)
    # This is a simple square polygon as an example
    custom_polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [-2.0, 51.0],
                [-2.0, 52.0],
                [-1.0, 52.0],
                [-1.0, 51.0],
                [-2.0, 51.0],
            ]
        ],
    }

    # Note: Check API documentation to see if polygon parameter is supported
    # If not, fetch with a broader area and use filter_by_polygon()

    # Example 4: Post-process filtering with polygon
    # If API doesn't support polygon directly, fetch broader area and filter
    try:
        filtered_data = api.filter_by_polygon(
            temp_data, polygon=custom_polygon["coordinates"][0]
        )
        logger.info(
            f"Filtered to {len(filtered_data)} observations within polygon"
        )
    except ImportError:
        logger.warning(
            "shapely not installed - skipping polygon filtering example"
        )

    # Export to CSV
    # combined_data.to_csv("ea_water_quality.csv", index=False)
