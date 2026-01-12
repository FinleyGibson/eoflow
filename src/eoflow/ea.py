"""
Environment Agency Water Quality Data API Client

A Python module to fetch water quality monitoring data from the Environment Agency's
Water Quality API. This module handles API pagination, supports multiple determinands,
and can filter data by geographic area using precanned area codes.

API Documentation:
- https://environment.data.gov.uk/water-quality-beta/api-docs
- https://gist.github.com/canwaf/2afa25fc6160efb25ac72b7acd60278d

Common Determinands:
- 0076: Temperature of Water
- 0077: Conductivity at 25°C
- 0180: Orthophosphate, reactive as P
- 6396: Turbidity (NTU)

Common Area Codes (precannedArea parameter):
- "environment_agency,DCS": Environment Agency DCS area
- "environment_agency,SWX": Southwest region (Devon, Cornwall, Somerset, Dorset)
- "local_authority,E06000002": Example local authority area

Author: Adapted from R code by Francis Rowney
Last Updated: 2025
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from eoflow.config import get_config
from eoflow.log_utils import get_logger

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
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generate a list of (month_start, month_end) tuples for a date range.

        Args:
            start_date: Start date (datetime object)
            end_date: End date (datetime object)

        Returns:
            List of tuples with (month_start, month_end) datetime objects
        """
        month_ranges = []
        current = start_date

        while current < end_date:
            month_start = current
            month_end = min(
                (current + relativedelta(months=1)) - timedelta(days=1),
                end_date,
            )
            month_ranges.append((month_start, month_end))
            current = month_end + timedelta(days=1)

        return month_ranges

    def _make_request(
        self,
        determinand: str,
        date_from: str,
        date_to: str,
        precanned_area: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Make a single API request and return the data as a DataFrame.

        Args:
            determinand: Determinand code (e.g., "0076")
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            precanned_area: Precanned area code (e.g., "environment_agency,DCS")
            limit: Maximum number of records to return

        Returns:
            DataFrame with the API response or None if request fails
        """
        if limit is None:
            limit = self.max_limit

        # Build query parameters for POST request
        params = {
            "determinand": determinand,
            "dateFrom": date_from,
            "dateTo": date_to,
            "limit": limit,
        }

        # Add precanned area if provided
        if precanned_area is not None:
            params["precannedArea"] = precanned_area

        headers = {
            "Accept": "text/csv",
            "API-Version": "1",
        }

        response = None
        try:
            logger.debug(
                f"Making API POST request for {date_from} to {date_to}, "
                f"determinand={determinand}, precannedArea={precanned_area}"
            )

            response = self.session.post(
                self.base_url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )

            # Log response details for debugging
            logger.debug(
                f"Response status: {response.status_code}, URL: {response.url}"
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

        except requests.exceptions.HTTPError as e:
            if response is not None:
                logger.error(
                    f"HTTP error {response.status_code} for {date_from} to {date_to}: {str(e)}"
                )
                logger.error(f"Response content: {response.text[:500]}")
                warnings.warn(
                    f"HTTP {response.status_code} error for {date_from} to {date_to}. "
                    f"Check logs for details."
                )
            else:
                logger.error(
                    f"HTTP error for {date_from} to {date_to}: {str(e)}"
                )
                warnings.warn(
                    f"HTTP error for {date_from} to {date_to}: {str(e)}"
                )
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
                  - "environment_agency,SWX" for Southwest region (includes Devon)
                  - "local_authority,E06000002" for local authority areas
            verbose: If True, print progress messages

        Returns:
            DataFrame with water quality observations

        Raises:
            ValueError: If no area is provided
        """
        # Convert string dates to datetime
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Validate area parameter
        if area is None:
            logger.error("No area provided")
            raise ValueError("'area' parameter must be provided")

        logger.info(
            f"Fetching data for determinand {determinand} from {start_date.strftime('%Y-%m-%d')} "
            f"to {end_date.strftime('%Y-%m-%d')} for area {area}"
        )

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
                precanned_area=area,
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
                cols_to_drop = [col for col in cols_to_drop if col in df.columns]
                df = df.drop(columns=cols_to_drop)

                df_list.append(df)

            # Add delay between determinand requests
            time.sleep(self.delay)

        if not df_list:
            logger.warning("No data retrieved for any determinands")
            warnings.warn("No data retrieved for any determinands")
            return pd.DataFrame()

        # Join dataframes on common columns
        if len(df_list) == 1:
            result = df_list[0]
        else:
            # Full outer join on common identifier columns
            result = df_list[0]
            for df in df_list[1:]:
                # Find columns common to both dataframes
                common_cols = list(set(result.columns) & set(df.columns))

                # Filter to essential columns for merging (exclude result columns)
                merge_cols = [col for col in common_cols if col not in
                             ["result", "Temp Water", "Cond @ 25C", "Orthophospht", "TurbidityNTU"]]

                if merge_cols:
                    result = result.merge(df, on=merge_cols, how="outer")
                else:
                    # If no common columns, concatenate instead
                    result = pd.concat([result, df], ignore_index=True)

        logger.info(f"Combined data shape: {result.shape}")
        return result

    def filter_by_polygon(
        self,
        df: pd.DataFrame,
        polygon: List[Tuple[float, float]],
        lat_col: str = "sample.samplingPoint.latitude",
        lon_col: str = "sample.samplingPoint.longitude",
    ) -> pd.DataFrame:
        """
        Filter observations by a geographic polygon.

        Requires shapely to be installed.

        Args:
            df: DataFrame with observation data
            polygon: List of (lon, lat) tuples defining a polygon
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            Filtered DataFrame with only observations within the polygon

        Raises:
            ImportError: If shapely is not installed
            ValueError: If unexpected data type is provided
        """
        try:
            from shapely.geometry import Point, Polygon as ShapelyPolygon
        except ImportError:
            raise ImportError("shapely is required for polygon filtering")

        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(df)}")
            raise ValueError("Input must be a pandas DataFrame")

        if df.empty:
            logger.warning("Input DataFrame is empty")
            return df

        # Create polygon
        polygon_shape = ShapelyPolygon(polygon)

        # Convert lat/lon columns to numeric
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

        # Filter points within polygon
        def point_in_polygon(row):
            try:
                if pd.isna(row[lat_col]) or pd.isna(row[lon_col]):
                    return False
                point = Point(row[lon_col], row[lat_col])
                return polygon_shape.contains(point)
            except Exception as e:
                logger.debug(f"Error checking point: {e}")
                return False

        mask = df.apply(point_in_polygon, axis=1)
        filtered_df = df[mask]

        logger.info(
            f"Filtered {len(df)} observations to {len(filtered_df)} within polygon"
        )
        return filtered_df


# Convenience function for simple use cases
def get_ea_water_quality(
    determinand: str,
    start_date: str,
    end_date: str,
    area: Optional[str] = None,
    verbose: bool = True,
    timeout: Optional[int] = None,
    delay: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convenience function to fetch EA water quality data with minimal setup.

    Uses configuration values from the config file for API settings.

    Args:
        determinand: Determinand code (e.g., "0076")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        area: Precanned area code (e.g., "environment_agency,DCS")
              For Devon: Use "environment_agency,SWX" (Southwest region)
        verbose: Print progress messages
        timeout: Request timeout in seconds (default: 60)
        delay: Delay between requests in seconds (default: 1.0)

    Returns:
        DataFrame with water quality data

    Example:
        >>> # Get temperature data for Southwest region (includes Devon)
        >>> df = get_ea_water_quality(
        ...     determinand="0076",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ...     area="environment_agency,SWX"
        ... )
    """
    logger.debug(f"Convenience function called for determinand {determinand}")
    api = EAWaterQualityAPI(timeout=timeout, delay=delay)
    return api.get_data(
        determinand=determinand,
        start_date=start_date,
        end_date=end_date,
        area=area,
        verbose=verbose,
    )


# Example usage
if __name__ == "__main__":
    import logging
    from pathlib import Path

    # create test logger
    logger = logging.getLogger(f"src{__file__.split("src")[-1]}: {__name__}")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    test_config = {
        "api": {
            "ea_base_url": "https://environment.data.gov.uk/water-quality/data/observation?",
            "ea_max_limit": 2500,
            "ea_api_delay": 0.5,
            "ea_api_timeout": 120,
            "max_retries": 5,
        }
    }

    # Example 1: Fetch data for a single determinand using precanned area
    api = EAWaterQualityAPI(
        base_url=test_config["api"]["ea_base_url"],
        max_limit=test_config["api"]["ea_max_limit"],
        delay=test_config["api"]["ea_api_delay"],
        timeout=test_config["api"]["ea_api_timeout"],
    )

    # Environment Agency DCS area
    temp_data = api.get_data(
        determinand="0076",  # Temperature of Water
        start_date="2024-01-01",
        end_date="2024-03-31",
        area="environment_agency,DCS",
    )

    logger.info(f"Retrieved {len(temp_data)} temperature observations")
    if not temp_data.empty:
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
        area="environment_agency,DCS",
    )

    logger.info(f"Combined data shape: {combined_data.shape}")

    # Example 3: Using convenience function
    df = get_ea_water_quality(
        determinand="0076",
        start_date="2024-01-01",
        end_date="2024-03-31",
        area="environment_agency,DCS",
        timeout=120,
    )


    logger.info(f"Retrieved {len(df)} observations using convenience function")
    if not df.empty:
        df.to_csv("ea_water_quality.csv", index=False)

    logger.info(f"Script {__file__} completed successfully!")
