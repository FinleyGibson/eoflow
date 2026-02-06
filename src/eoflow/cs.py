"""
Citizen Scientist (CS) Water Quality Data API Client

A Python module to fetch citizen scientist water quality monitoring data from
ArcGIS FeatureServer APIs. This module handles spatial queries, supports filtering
by polygon areas, and provides convenient access to community-collected water quality data.

Example ArcGIS REST API endpoints:
- https://services1.arcgis.com/[ID]/arcgis/rest/services/[ServiceName]/FeatureServer/0

The data accessed through this API represents citizen scientist observations and
community monitoring efforts, providing valuable supplementary information to
official monitoring programs.

Author: Generated for eoflow project
Last Updated: 2025
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from eoflow.config import get_config
from eoflow.log_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class CSDataAPI:
    """
    Client for fetching citizen scientist data from ArcGIS FeatureServer APIs.

    This class provides methods to query ArcGIS REST services, filter data by
    geographic area, and convert results to pandas DataFrames for analysis.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        layer_id: Optional[int] = None,
        delay: Optional[float] = None,
        timeout: Optional[int] = None,
        max_records: Optional[int] = None,
    ):
        """
        Initialize the CS Data API client.

        Args:
            base_url: ArcGIS FeatureServer base URL. If None, uses config value.
            layer_id: Layer ID within the FeatureServer (default: 0)
            delay: Time to wait between API requests (seconds). If None, uses config value.
            timeout: Request timeout in seconds. If None, uses config value.
            max_records: Maximum records per request. If None, uses config value.
        """
        config = get_config()

        self.base_url = (
            base_url if base_url is not None else config.get("api.cs_base_url")
        )
        self.layer_id = (
            layer_id
            if layer_id is not None
            else config.get("api.cs_layer_id", 0)
        )
        self.delay = (
            delay if delay is not None else config.get("api.cs_api_delay", 1.0)
        )
        self.timeout = (
            timeout
            if timeout is not None
            else config.get("api.cs_api_timeout", 30)
        )
        self.max_records = (
            max_records
            if max_records is not None
            else config.get("api.cs_max_records", 2000)
        )

        self.session = requests.Session()
        self.query_url = f"{self.base_url}/{self.layer_id}/query"

        logger.debug(
            f"Initialized CSDataAPI with base_url={self.base_url}, "
            f"layer_id={self.layer_id}, delay={self.delay}s, "
            f"timeout={self.timeout}s, max_records={self.max_records}"
        )

    def _make_request(
        self,
        params: Dict[str, Any],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Make a request to the ArcGIS FeatureServer API.

        Args:
            params: Query parameters for the API request
            verbose: If True, print detailed request information

        Returns:
            JSON response as a dictionary

        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the API returns an error response
        """
        # Add default parameters for ArcGIS REST API
        default_params = {
            "f": "json",  # Response format
            "outFields": "*",  # Return all fields
            "returnGeometry": "true",  # Include geometry
            "spatialRel": "esriSpatialRelIntersects",
        }

        # Merge with user params (user params take precedence)
        full_params = {**default_params, **params}

        if verbose:
            logger.info(f"Making request to: {self.query_url}")
            logger.info(f"Parameters: {json.dumps(full_params, indent=2)}")

        try:
            response = self.session.get(
                self.query_url,
                params=full_params,
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()

            # Check for ArcGIS API errors
            if "error" in data:
                error_msg = data["error"].get("message", "Unknown error")
                error_details = data["error"].get("details", [])
                raise ValueError(
                    f"ArcGIS API error: {error_msg}. Details: {error_details}"
                )

            if verbose:
                feature_count = len(data.get("features", []))
                logger.info(f"Retrieved {feature_count} features")

            # Rate limiting
            if self.delay > 0:
                time.sleep(self.delay)

            return data

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout} seconds")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    def get_data(
        self,
        where: str = "1=1",
        geometry: Optional[str] = None,
        geometry_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_field: str = "sample_date",
        out_fields: str = "*",
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Get CS data from the ArcGIS FeatureServer.

        Args:
            where: SQL WHERE clause for filtering (default: "1=1" returns all)
            geometry: Geometry filter (e.g., polygon coordinates as JSON string)
            geometry_type: Type of geometry filter (e.g., "esriGeometryPolygon")
            start_date: Start date for filtering (format: "YYYY-MM-DD")
            end_date: End date for filtering (format: "YYYY-MM-DD")
            date_field: Name of the date field to filter on
            out_fields: Comma-separated list of fields to return (default: "*")
            verbose: If True, print detailed information

        Returns:
            DataFrame with CS data

        Raises:
            ValueError: If the request fails or returns invalid data
        """
        # Build WHERE clause with date filtering if specified
        where_clause = where
        if start_date and end_date:
            # Convert dates to timestamps for ArcGIS
            start_ts = int(
                datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000
            )
            end_ts = int(
                datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000
            )
            date_filter = (
                f"{date_field} >= {start_ts} AND {date_field} <= {end_ts}"
            )

            if where_clause and where_clause != "1=1":
                where_clause = f"({where_clause}) AND ({date_filter})"
            else:
                where_clause = date_filter

        params = {
            "where": where_clause,
            "outFields": out_fields,
            "resultRecordCount": self.max_records,
        }

        # Add geometry filter if provided
        if geometry and geometry_type:
            params["geometry"] = geometry
            params["geometryType"] = geometry_type

        if verbose:
            logger.info(f"Fetching CS data with WHERE clause: {where_clause}")

        try:
            response = self._make_request(params, verbose=verbose)

            features = response.get("features", [])

            if not features:
                logger.warning("No CS data returned from API")
                return pd.DataFrame()

            # Extract attributes from features
            records = []
            for feature in features:
                record = feature.get("attributes", {})

                # Add geometry information if present
                if "geometry" in feature and feature["geometry"]:
                    geom = feature["geometry"]
                    if "x" in geom and "y" in geom:
                        record["longitude"] = geom["x"]
                        record["latitude"] = geom["y"]
                    elif "rings" in geom or "paths" in geom:
                        # For polygon/polyline geometries, store as JSON string
                        record["geometry_json"] = json.dumps(geom)

                records.append(record)

            df = pd.DataFrame(records)

            # Convert timestamp fields to datetime
            for col in df.columns:
                if df[col].dtype == "int64" and col.lower().endswith(
                    ("date", "time", "_dt")
                ):
                    try:
                        # ArcGIS uses milliseconds since epoch
                        df[col] = pd.to_datetime(df[col], unit="ms")
                    except Exception:
                        # Conversion failed, leave column as-is
                        pass

            logger.info(f"Retrieved {len(df)} CS records")

            return df

        except Exception as e:
            logger.error(f"Error fetching CS data: {str(e)}")
            raise

    def get_data_by_bbox(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        spatial_reference: int = 4326,
        where: str = "1=1",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_field: str = "sample_date",
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Get CS data within a bounding box.

        Args:
            xmin: Minimum longitude/x coordinate
            ymin: Minimum latitude/y coordinate
            xmax: Maximum longitude/x coordinate
            ymax: Maximum latitude/y coordinate
            spatial_reference: WKID for spatial reference (default: 4326 for WGS84)
            where: Additional SQL WHERE clause for filtering
            start_date: Start date for filtering (format: "YYYY-MM-DD")
            end_date: End date for filtering (format: "YYYY-MM-DD")
            date_field: Name of the date field to filter on
            verbose: If True, print detailed information

        Returns:
            DataFrame with CS data within the bounding box
        """
        # Construct envelope geometry
        geometry = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "spatialReference": {"wkid": spatial_reference},
        }

        geometry_str = json.dumps(geometry)

        return self.get_data(
            where=where,
            geometry=geometry_str,
            geometry_type="esriGeometryEnvelope",
            start_date=start_date,
            end_date=end_date,
            date_field=date_field,
            verbose=verbose,
        )

    def filter_by_polygon(
        self,
        df: pd.DataFrame,
        polygon: List[Tuple[float, float]],
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> pd.DataFrame:
        """
        Filter DataFrame to only include points within a polygon.

        Args:
            df: DataFrame with CS data
            polygon: List of (lon, lat) tuples defining the polygon boundary
            lat_col: Name of the latitude column
            lon_col: Name of the longitude column

        Returns:
            Filtered DataFrame with only points inside the polygon

        Raises:
            ImportError: If shapely is not installed
            ValueError: If required columns are missing
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to filter_by_polygon")
            return df

        # Check if required columns exist
        if lat_col not in df.columns or lon_col not in df.columns:
            logger.warning(
                f"Required columns '{lat_col}' and/or '{lon_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
            return df

        try:
            from shapely.geometry import Point, Polygon

            # Create Shapely polygon
            poly = Polygon(polygon)

            def point_in_polygon(row):
                """Check if a point is inside the polygon."""
                try:
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])

                    # Skip invalid coordinates
                    if pd.isna(lat) or pd.isna(lon):
                        return False

                    point = Point(lon, lat)
                    return poly.contains(point)
                except (ValueError, TypeError):
                    return False

            # Filter DataFrame
            mask = df.apply(point_in_polygon, axis=1)
            filtered_df = df.loc[mask].copy()

            logger.info(
                f"Filtered {len(df)} records to {len(filtered_df)} records "
                f"within polygon"
            )

            return filtered_df

        except ImportError:
            logger.error(
                "shapely is required for polygon filtering. "
                "Install with: pip install shapely"
            )
            raise ImportError(
                "shapely is required for polygon filtering. "
                "Please install it with: pip install shapely"
            )

    def get_layer_info(self) -> Dict[str, Any]:
        """
        Get metadata information about the FeatureServer layer.

        Returns:
            Dictionary containing layer metadata (fields, geometry type, etc.)
        """
        info_url = f"{self.base_url}/{self.layer_id}"

        try:
            response = self.session.get(
                info_url,
                params={"f": "json"},
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()

            if "error" in data:
                raise ValueError(f"ArcGIS API error: {data['error']}")

            logger.info(
                f"Retrieved layer info for: {data.get('name', 'Unknown')}"
            )

            return data

        except Exception as e:
            logger.error(f"Error fetching layer info: {str(e)}")
            raise

    def get_unique_values(
        self,
        field: str,
        where: str = "1=1",
        verbose: bool = False,
    ) -> List[Any]:
        """
        Get unique values for a specific field.

        Args:
            field: Field name to get unique values for
            where: SQL WHERE clause for filtering
            verbose: If True, print detailed information

        Returns:
            List of unique values
        """
        # Use a query with DISTINCT or get all data and extract unique values
        df = self.get_data(
            where=where,
            out_fields=field,
            verbose=verbose,
        )

        if df.empty or field not in df.columns:
            return []

        unique_vals = df[field].dropna().unique().tolist()

        logger.info(
            f"Found {len(unique_vals)} unique values for field '{field}'"
        )

        return unique_vals


def get_cs_data(
    base_url: str,
    layer_id: int = 0,
    where: str = "1=1",
    polygon: Optional[List[Tuple[float, float]]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_field: str = "sample_date",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to fetch CS data from an ArcGIS FeatureServer.

    Args:
        base_url: ArcGIS FeatureServer base URL
        layer_id: Layer ID within the FeatureServer (default: 0)
        where: SQL WHERE clause for filtering (default: "1=1" returns all)
        polygon: Optional list of (lon, lat) tuples for polygon filtering
        bbox: Optional bounding box as (xmin, ymin, xmax, ymax)
        start_date: Start date for filtering (format: "YYYY-MM-DD")
        end_date: End date for filtering (format: "YYYY-MM-DD")
        date_field: Name of the date field to filter on
        verbose: If True, print detailed information

    Returns:
        DataFrame with CS data

    Example:
        >>> df = get_cs_data(
        ...     base_url="https://services.arcgis.com/xxx/arcgis/rest/services/CSData/FeatureServer",
        ...     layer_id=0,
        ...     bbox=(-4.5, 50.0, -3.0, 51.0),
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ...     verbose=True
        ... )
    """
    api = CSDataAPI(base_url=base_url, layer_id=layer_id)

    if bbox:
        # Use bounding box query
        xmin, ymin, xmax, ymax = bbox
        df = api.get_data_by_bbox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            where=where,
            start_date=start_date,
            end_date=end_date,
            date_field=date_field,
            verbose=verbose,
        )
    else:
        # Regular query
        df = api.get_data(
            where=where,
            start_date=start_date,
            end_date=end_date,
            date_field=date_field,
            verbose=verbose,
        )

    # Apply polygon filtering if provided
    if polygon and not df.empty:
        df = api.filter_by_polygon(df, polygon)

    return df
