from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, mapping
from shapely.geometry.base import BaseGeometry

from eoflow.log_utils import get_logger

if TYPE_CHECKING:
    import openeo  # type: ignore[import-untyped]
    import xarray  # type: ignore[import-untyped]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default openEO backend URL (Copernicus Data Space Federation)
OPENEO_BACKEND = "openeofed.dataspace.copernicus.eu"

#: Default Sentinel-2 L2A collection on that backend
SENTINEL2_COLLECTION = "SENTINEL2_L2A"

#: Band names for common spectral indices on SENTINEL2_L2A
_S2_BANDS: Dict[str, Dict[str, str]] = {
    "NDVI": {"nir": "B08", "red": "B04"},
    "NDWI": {"green": "B03", "nir": "B08"},
    "EVI": {"nir": "B08", "red": "B04", "blue": "B02"},
    "MNDWI": {"green": "B03", "swir": "B11"},
    "NDRE": {"rededge": "B05", "red": "B04"},
}

# Columns written by delineate_catchments that we care about
_COL_LAT = "latitude"
_COL_LON = "longitude"
_COL_SNAP_LAT = "snap_latitude"
_COL_SNAP_LON = "snap_longitude"
_COL_FLOW_ACC = "flow_acc_at_pour_point"
_COL_STATUS = "delineation_status"
_COL_ERROR = "delineation_error"
_COL_RESULT = "result"
_COL_DATE = "Date"
_COL_SITE = "samplingPoint.prefLabel"
_COL_NOTATION = "samplingPoint.notation"
_COL_WKT = "__catchment_wkt"


# ---------------------------------------------------------------------------
# CatchmentLayers
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class CatchmentLayers:
    """Container for all computed spatial layers associated with a catchment.

    Each attribute is populated by the corresponding ``compute_*`` method on
    :class:`Sample` and is ``None`` until computed.  This dataclass
    is intentionally kept separate from :class:`Sample` so that
    computed results — some of which carry a temporal dimension — can be
    inspected, serialised, or passed around independently of the observation
    metadata.

    Attributes
    ----------
    topography : xarray.DataArray or None
        Elevation in metres on a BNG (EPSG:27700) regular grid, shape
        ``(y, x)``.  Populated by :meth:`Sample.compute_topography`.
    slope : xarray.DataArray or None
        Terrain slope grid (units depend on the ``output`` kwarg supplied to
        :meth:`Sample.compute_slope`, default degrees), shape
        ``(y, x)``.  Populated by :meth:`Sample.compute_slope`.
    aspect : xarray.DataArray or None
        Terrain aspect (slope direction) as a compass bearing clockwise from
        North, shape ``(y, x)``.  Units depend on the ``output`` kwarg
        supplied to :meth:`Sample.compute_aspect` (default degrees,
        range 0–360).  Populated by :meth:`Sample.compute_aspect`.
    soil_type : pandas.Series or None
        Fractional SEARG soil-group coverage within the catchment (values
        0–1, indexed by the 12 SEARG group names).  Values sum to ≤ 1 (the
        gap represents area not covered by any SEARG polygon, e.g. Scotland
        or the sea).  Populated by :meth:`Sample.compute_soil_type`.
    soil_polygons : geopandas.GeoDataFrame or None
        Raw SEARG soil polygons (in EPSG:27700) that intersect the catchment,
        as returned by :func:`eoflow.soil.query_soil_polygons`.  Key columns:
        ``SEARG_Concise``, ``MU_NAME``, ``BFI``, ``SPR``,
        ``SEARGDescription``.  Populated alongside :attr:`soil_type` by
        :meth:`Sample.compute_soil_type`.
    rainfall : xarray.DataArray or None
        Rainfall rate in mm/h with dimensions ``(time, y, x)`` on a BNG
        grid.  Unlike the static layers, this carries an explicit temporal
        dimension — use the ``start``/``end`` parameters of
        :meth:`Sample.compute_rainfall` to control the time window.
        Populated by :meth:`Sample.compute_rainfall`.
    """

    topography: Optional["xarray.DataArray"] = field(default=None)
    slope: Optional["xarray.DataArray"] = field(default=None)
    aspect: Optional["xarray.DataArray"] = field(default=None)
    soil_type: Optional[pd.Series] = field(default=None)
    soil_polygons: Optional[gpd.GeoDataFrame] = field(default=None)
    rainfall: Optional["xarray.DataArray"] = field(default=None)
    ndvi: Optional["xarray.DataArray"] = field(default=None)
    """NDVI time series (t, y, x).  Populated by :meth:`~Sample.fetch_ndvi`."""
    ndwi: Optional["xarray.DataArray"] = field(default=None)
    """NDWI time series (t, y, x).  Populated by :meth:`~Sample.fetch_ndwi`."""
    eo_bands: Optional["xarray.Dataset"] = field(default=None)
    """Multi-band EO Dataset.  Populated by :meth:`~Sample.fetch_bands`."""

    @property
    def available(self) -> List[str]:
        """Names of layers that have been computed (non-``None``)."""
        return [
            name
            for name in (
                "topography",
                "slope",
                "aspect",
                "soil_type",
                "soil_polygons",
                "rainfall",
                "ndvi",
                "ndwi",
                "eo_bands",
            )
            if getattr(self, name) is not None
        ]

    def __repr__(self) -> str:
        available = self.available
        if available:
            return f"CatchmentLayers(computed={available!r})"
        return "CatchmentLayers(no layers computed yet)"


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


class Sample:
    """A single water-quality sampling point with optional catchment geometry.

    Wraps one row from the GeoPackage produced by ``delineate_catchments.py`` and
    exposes typed property accessors for the most-used fields (site name,
    notation, coordinates, measurement result, date …).  When catchment
    delineation has been run the polygon is stored in :attr:`catchment` and
    derived spatial helpers (:meth:`catchment_bbox`, :meth:`catchment_geojson`,
    :meth:`catchment_area_km2`) become available.

    Earth-observation layers (NDVI, NDWI, band composites) are fetched on
    demand via the ``fetch_*`` family of methods and stored in
    :attr:`layers`.

    Parameters
    ----------
    row : pandas.Series
        A single row from the catchment GeoPackage.
    catchment : shapely geometry or None
        The delineated catchment polygon, or *None* if delineation failed or
        has not been run yet.
    """

    def __init__(self, row: pd.Series, catchment: Optional[BaseGeometry]) -> None:
        self._row = row.copy()
        self.catchment: Optional[BaseGeometry] = catchment
        self.layers: CatchmentLayers = CatchmentLayers()

    # ------------------------------------------------------------------
    # Convenience property accessors
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        return str(self._row.get("id", ""))

    @property
    def site_name(self) -> str:
        return str(self._row.get(_COL_SITE, ""))

    @property
    def notation(self) -> str:
        return str(self._row.get(_COL_NOTATION, ""))

    @property
    def latitude(self) -> Optional[float]:
        v = self._row.get(_COL_LAT)
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None

    @property
    def longitude(self) -> Optional[float]:
        v = self._row.get(_COL_LON)
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None

    @property
    def sample_point(self) -> Optional[Point]:
        """Shapely Point (lon, lat) for the original sampling location."""
        if self.longitude is not None and self.latitude is not None:
            return Point(self.longitude, self.latitude)
        return None

    @property
    def snap_latitude(self) -> Optional[float]:
        v = self._row.get(_COL_SNAP_LAT)
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None

    @property
    def snap_longitude(self) -> Optional[float]:
        v = self._row.get(_COL_SNAP_LON)
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None

    @property
    def snapped_pour_point(self) -> Optional[Point]:
        """Shapely Point for the snapped pour point on the stream network."""
        if self.snap_longitude is not None and self.snap_latitude is not None:
            return Point(self.snap_longitude, self.snap_latitude)
        return None

    @property
    def flow_acc_at_pour_point(self) -> Optional[float]:
        v = self._row.get(_COL_FLOW_ACC)
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None

    @property
    def result(self) -> Optional[float]:
        v = self._row.get(_COL_RESULT)
        return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None

    @property
    def unit(self) -> str:
        return str(self._row.get("unit", ""))

    @property
    def determinand(self) -> str:
        return str(self._row.get("determinand.prefLabel", ""))

    @property
    def date(self) -> Optional[date]:
        raw = self._row.get(_COL_DATE)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return None
        try:
            return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
        except ValueError:
            return None

    @property
    def delineation_status(self) -> str:
        return str(self._row.get(_COL_STATUS, ""))

    @property
    def delineation_error(self) -> str:
        return str(self._row.get(_COL_ERROR, ""))

    @property
    def has_catchment(self) -> bool:
        poly = self.catchment
        return poly is not None and not poly.is_empty

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    def catchment_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Return the WGS 84 bounding box of the catchment as
        ``(west, south, east, north)``, or *None* if no catchment."""
        poly = self.catchment
        if poly is None or poly.is_empty:
            return None
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        return bounds  # (west, south, east, north)

    def catchment_geojson(self) -> Optional[Dict[str, Any]]:
        """Return the catchment as a GeoJSON-compatible mapping dict."""
        poly = self.catchment
        if poly is None or poly.is_empty:
            return None
        return mapping(poly)  # type: ignore[return-value]

    def catchment_area_km2(self) -> Optional[float]:
        """Approximate catchment area in km².

        Uses the local equirectangular approximation around the catchment
        centroid — adequate for the small polygons typical of individual
        stream catchments.
        """
        poly = self.catchment
        if poly is None or poly.is_empty:
            return None
        centroid = poly.centroid
        lat_rad = math.radians(centroid.y)
        # Degrees to metres at this latitude
        metres_per_deg_lat = 111_132.0
        metres_per_deg_lon = 111_320.0 * math.cos(lat_rad)
        # Polygon area in square degrees, scale to km²
        area_deg2 = poly.area
        area_km2 = area_deg2 * metres_per_deg_lat * metres_per_deg_lon / 1e6
        return area_km2

    # ------------------------------------------------------------------
    # Catchment delineation
    # ------------------------------------------------------------------

    def delineate(
        self,
        dem_path: Union[str, Path],
        *,
        flow_acc_threshold: int = 5000,
        **kwargs: Any,
    ) -> None:
        """Delineate the catchment polygon from a DEM for this sample's pour point.

        Wraps :func:`eoflow.catchment.delineate_catchment_with_metadata` and
        updates :attr:`catchment`, the snapped-pour-point row fields, the
        flow-accumulation value, and :attr:`delineation_status` in place.

        Parameters
        ----------
        dem_path : str or Path
            Path to a GeoTIFF DEM covering the pour-point location.
        flow_acc_threshold : int
            Minimum upstream cell count used when snapping the pour point to
            the nearest stream.  Higher values snap to larger streams
            (default: 5000).
        **kwargs
            Additional keyword arguments forwarded to
            :func:`~eoflow.catchment.delineate_catchment_with_metadata`
            (e.g. ``routing``, ``pit_fill``, ``fill_depressions``).

        Raises
        ------
        ValueError
            If the sample has no recorded latitude/longitude coordinates.
        FileNotFoundError
            If *dem_path* does not exist.

        Notes
        -----
        On success the following are updated in place:

        * ``self.catchment`` — the new :class:`~shapely.geometry.Polygon`
        * ``self._row["snap_latitude"]`` / ``"snap_longitude"``
        * ``self._row["flow_acc_at_pour_point"]``
        * ``self._row["delineation_status"]`` → ``"ok"``
        * ``self._row["delineation_error"]`` → ``""``

        On failure a warning is logged, ``delineation_status`` is set to
        ``"error"``, and the exception is re-raised.
        """
        from eoflow.catchment import delineate_catchment_with_metadata

        point = self.sample_point
        if point is None:
            raise ValueError(
                f"Sample '{self.id}' has no recorded coordinates — cannot delineate a catchment."
            )

        logger.info("Delineating catchment for site '%s' …", self.site_name)
        try:
            meta = delineate_catchment_with_metadata(
                point,
                dem_path,
                flow_acc_threshold=flow_acc_threshold,
                **kwargs,
            )
        except Exception as exc:
            self._row[_COL_STATUS] = "error"
            self._row[_COL_ERROR] = str(exc)
            logger.warning(
                "Catchment delineation failed for site '%s': %s",
                self.site_name,
                exc,
            )
            raise

        self.catchment = meta["polygon"]
        snapped: Point = meta["snapped_pour_point"]
        self._row[_COL_SNAP_LAT] = snapped.y
        self._row[_COL_SNAP_LON] = snapped.x
        self._row[_COL_FLOW_ACC] = meta["flow_acc_at_pour_point"]
        self._row[_COL_STATUS] = "ok"
        self._row[_COL_ERROR] = ""

        logger.info(
            "Catchment delineated for site '%s' — area ≈ %.3f km².",
            self.site_name,
            self.catchment_area_km2() or float("nan"),
        )

    # ------------------------------------------------------------------
    # Spatial layer computation
    # ------------------------------------------------------------------

    def compute_topography(
        self,
        dem_path: Union[str, Path],
        *,
        res: int = 50,
    ) -> "xarray.DataArray":
        """Extract elevation data for the catchment polygon from a DEM.

        Delegates to :func:`eoflow.topography.get_topography_for_polygon` and
        stores the result in :attr:`layers.topography`.

        Parameters
        ----------
        dem_path : str or Path
            Path to a GeoTIFF DEM (any CRS; reprojected internally to BNG).
        res : int
            Output grid resolution in BNG metres (default: 50 m).

        Returns
        -------
        xarray.DataArray
            Elevation in metres with dimensions ``(y, x)``, CRS EPSG:27700.
            Also stored in ``self.layers.topography``.

        Raises
        ------
        ValueError
            If the sample has no delineated catchment polygon.
        FileNotFoundError
            If *dem_path* does not exist.
        """
        from eoflow.topography import get_topography_for_polygon

        if not self.has_catchment:
            raise ValueError(
                f"Sample '{self.id}' has no catchment polygon. "
                "Run delineate() first, or load a dataset that includes catchments."
            )

        logger.info("Computing topography for site '%s' …", self.site_name)
        da = get_topography_for_polygon(self.catchment, dem_path, res=res)
        self.layers.topography = da
        logger.info(
            "Topography stored: grid %d × %d, elevation range %.1f – %.1f m.",
            da.sizes.get("y", 0),
            da.sizes.get("x", 0),
            float(da.min()),
            float(da.max()),
        )
        return da

    def compute_slope(
        self,
        dem_path: Union[str, Path],
        *,
        res: int = 50,
        output: Literal["degrees", "percent_rise", "radians"] = "degrees",
    ) -> "xarray.DataArray":
        """Compute terrain slope for the catchment polygon from a DEM.

        Delegates to :func:`eoflow.topography.get_slope_for_polygon` using
        Horn's (1981) 3 × 3 neighbourhood method and stores the result in
        :attr:`layers.slope`.

        Parameters
        ----------
        dem_path : str or Path
            Path to a GeoTIFF DEM (any CRS; reprojected internally to BNG).
        res : int
            Output grid resolution in BNG metres (default: 50 m).
        output : Literal["degrees", "percent_rise", "radians"]
            Slope units: ``"degrees"`` (default), ``"percent_rise"``, or
            ``"radians"``.

        Returns
        -------
        xarray.DataArray
            Slope grid in the requested units with dimensions ``(y, x)``,
            CRS EPSG:27700.  Also stored in ``self.layers.slope``.

        Raises
        ------
        ValueError
            If the sample has no delineated catchment polygon, or if *output*
            is not a recognised unit string.
        FileNotFoundError
            If *dem_path* does not exist.
        """
        from eoflow.topography import get_slope_for_polygon

        if not self.has_catchment:
            raise ValueError(
                f"Sample '{self.id}' has no catchment polygon. "
                "Run delineate() first, or load a dataset that includes catchments."
            )

        logger.info("Computing slope (%s) for site '%s' …", output, self.site_name)
        da = get_slope_for_polygon(self.catchment, dem_path, res=res, output=output)
        self.layers.slope = da
        logger.info(
            "Slope stored: grid %d × %d, range %.2f – %.2f %s.",
            da.sizes.get("y", 0),
            da.sizes.get("x", 0),
            float(da.min()),
            float(da.max()),
            da.attrs.get("units", output),
        )
        return da

    def compute_aspect(
        self,
        dem_path: Union[str, Path],
        *,
        res: int = 50,
        output: Literal["degrees", "radians"] = "degrees",
    ) -> "xarray.DataArray":
        """Compute terrain aspect (slope direction) for the catchment from a DEM.

        Delegates to :func:`eoflow.topography.get_aspect_for_polygon` using
        Horn's (1981) 3 × 3 neighbourhood method and stores the result in
        :attr:`layers.aspect`.

        Aspect is measured **clockwise from North** (0 = North, 90 = East,
        180 = South, 270 = West).  Flat cells and border cells that lack a
        full neighbourhood are ``NaN``.

        Parameters
        ----------
        dem_path : str or Path
            Path to a GeoTIFF DEM (any CRS; reprojected internally to BNG).
        res : int
            Output grid resolution in BNG metres (default: 50 m).
        output : Literal["degrees", "radians"]
            Aspect units: ``"degrees"`` (default, range 0–360) or
            ``"radians"`` (range 0–2π).

        Returns
        -------
        xarray.DataArray
            Aspect grid in the requested units with dimensions ``(y, x)``,
            CRS EPSG:27700.  Also stored in ``self.layers.aspect``.

        Raises
        ------
        ValueError
            If the sample has no delineated catchment polygon, or if *output*
            is not a recognised unit string.
        FileNotFoundError
            If *dem_path* does not exist.
        """
        from eoflow.topography import get_aspect_for_polygon

        if not self.has_catchment:
            raise ValueError(
                f"Sample '{self.id}' has no catchment polygon. "
                "Run delineate() first, or load a dataset that includes catchments."
            )

        logger.info("Computing aspect (%s) for site '%s' …", output, self.site_name)
        da = get_aspect_for_polygon(self.catchment, dem_path, res=res, output=output)
        self.layers.aspect = da
        logger.info(
            "Aspect stored: grid %d × %d, range %.2f – %.2f %s.",
            da.sizes.get("y", 0),
            da.sizes.get("x", 0),
            float(da.min()),
            float(da.max()),
            da.attrs.get("units", output),
        )
        return da

    def compute_soil_type(
        self,
        *,
        session: Optional[Any] = None,
        timeout: int = 60,
    ) -> pd.Series:
        """Query SEARG soil-group fractional coverage for the catchment.

        Delegates to :func:`eoflow.soil.soil_coverage` and stores the result
        in :attr:`layers.soil_type`.

        Parameters
        ----------
        session : requests.Session, optional
            Optional reusable HTTP session for connection pooling or retries.
            A new session is created internally if not provided.
        timeout : int
            HTTP request timeout in seconds (default: 60).

        Returns
        -------
        pandas.Series
            Fractional SEARG soil-group coverage (0–1), indexed by the 12
            SEARG group names.  Values sum to ≤ 1.
            Also stored in ``self.layers.soil_type``.

        Raises
        ------
        ValueError
            If the sample has no delineated catchment polygon.
        requests.HTTPError
            On a non-2xx response from the SEARG ArcGIS service.
        RuntimeError
            If the SEARG service returns an error payload.

        Notes
        -----
        Coverage is computed for England and Wales only.  Catchments that
        extend into Scotland, Ireland, or offshore will show lower total
        fractional coverage.
        """
        from eoflow.soil import query_soil_polygons, soil_coverage

        if not self.has_catchment:
            raise ValueError(
                f"Sample '{self.id}' has no catchment polygon. "
                "Run delineate() first, or load a dataset that includes catchments."
            )

        logger.info("Querying SEARG soil type for site '%s' …", self.site_name)
        gdf = query_soil_polygons(self.catchment, session=session, timeout=timeout)
        self.layers.soil_polygons = gdf
        series = soil_coverage(self.catchment, session=session, timeout=timeout, polygons_gdf=gdf)
        self.layers.soil_type = series
        n_groups = int((series > 0).sum())
        logger.info(
            "Soil type stored: %d SEARG group(s) present, total coverage %.1f%%.",
            n_groups,
            series.sum() * 100,
        )
        return series

    def compute_rainfall(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        *,
        nimrod_dir: Union[str, Path],
    ) -> "xarray.DataArray":
        """Load and extract NIMROD 1 km composite rainfall data for the catchment.

        Delegates to :func:`eoflow.rainfall.get_nimrod_rainfall_for_polygon`
        to load pre-processed NIMROD data from disk and stores the result in
        :attr:`layers.rainfall`.

        Unlike the static layers (topography, slope, soil type), rainfall
        data carries an explicit temporal dimension: the returned array has
        shape ``(time, y, x)`` where ``time`` holds UTC timestamps.

        Parameters
        ----------
        start : str or datetime
            Start of the time window (UTC, inclusive).  Either a
            ``"YYYY-MM-DDTHH:MM"`` / ISO-8601 string or an aware
            :class:`~datetime.datetime`.
        end : str or datetime
            End of the time window (UTC, inclusive).
        nimrod_dir : str or Path
            Path to the NIMROD data on disk.  Either:

            * A **directory** of per-timestep NetCDF files in the layout
              produced by ``scripts/process_nimrod_local.py``::

                  <nimrod_dir>/{year}/{YYYYMMDD}/{YYYYMMDD_HHMMSS}.nc

            * A **single consolidated** ``.nc`` file produced by
              ``scripts/consolidate_nimrod.py``.

        Returns
        -------
        xarray.DataArray
            Rainfall rate in mm/h with dimensions ``(time, y, x)``, CRS
            EPSG:27700.  Also stored in ``self.layers.rainfall``.

        Raises
        ------
        ValueError
            If the sample has no delineated catchment polygon, or if *end*
            precedes *start*.
        """
        from eoflow.rainfall import get_nimrod_rainfall_for_polygon

        if not self.has_catchment:
            raise ValueError(
                f"Sample '{self.id}' has no catchment polygon. "
                "Run delineate() first, or load a dataset that includes catchments."
            )

        logger.info(
            "Fetching NIMROD rainfall for site '%s' (%s → %s) …",
            self.site_name,
            start,
            end,
        )
        da = get_nimrod_rainfall_for_polygon(
            self.catchment,
            start,
            end,
            nimrod_dir=nimrod_dir,
        )
        self.layers.rainfall = da
        n_times = da.sizes.get("time", 0)
        logger.info(
            "Rainfall stored: %d timestep(s), grid %d × %d.",
            n_times,
            da.sizes.get("y", 0),
            da.sizes.get("x", 0),
        )
        return da

    def _compute_rainfall_metoffice_legacy(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        *,
        download_dir: Optional[Union[str, Path]] = None,
        res: int = 1000,
        run_hour: Optional[int] = None,
        workers: int = 4,
    ) -> "xarray.DataArray":
        """[Legacy] Download and extract Met Office UKV rainfall-rate data for the catchment.

        .. deprecated::
            Use :meth:`compute_rainfall` instead, which loads pre-processed
            NIMROD 1 km composite data from disk rather than downloading
            from the Met Office AWS S3 bucket.

        Delegates to :func:`eoflow.rainfall.get_rainfall_for_polygon` and
        stores the result in :attr:`layers.rainfall`.

        Unlike the static layers (topography, slope, soil type), rainfall
        data carries an explicit temporal dimension: the returned array has
        shape ``(time, y, x)`` where ``time`` holds UTC timestamps at
        15-minute intervals.

        Parameters
        ----------
        start : str or datetime
            Start of the time window (UTC, inclusive).  Either a
            ``"YYYY-MM-DDTHH:MM"`` string or an aware
            :class:`~datetime.datetime`.
        end : str or datetime
            End of the time window (UTC, inclusive).
        download_dir : str or Path, optional
            Directory for caching downloaded ``*.nc`` files.  Pass an
            explicit path to avoid re-downloading on repeated calls.  When
            ``None``, a temporary directory is used and cleaned up
            automatically.
        res : int
            Output BNG grid resolution in metres (default: 1000 m).
        run_hour : int or None
            Restrict to files from the model run starting at this UTC hour
            (0–23).  ``None`` (default) uses all available runs.
        workers : int
            Number of parallel download threads (default: 4).

        Returns
        -------
        xarray.DataArray
            Rainfall rate in mm/h with dimensions ``(time, y, x)``, CRS
            EPSG:27700.  Also stored in ``self.layers.rainfall``.

        Raises
        ------
        ValueError
            If the sample has no delineated catchment polygon, or if *end*
            precedes *start*.
        """
        from eoflow.rainfall import get_rainfall_for_polygon

        if not self.has_catchment:
            raise ValueError(
                f"Sample '{self.id}' has no catchment polygon. "
                "Run delineate() first, or load a dataset that includes catchments."
            )

        logger.info(
            "Fetching rainfall for site '%s' (%s → %s) …",
            self.site_name,
            start,
            end,
        )
        da = get_rainfall_for_polygon(
            self.catchment,
            start,
            end,
            download_dir=download_dir,
            res=res,
            run_hour=run_hour,
            workers=workers,
        )
        self.layers.rainfall = da
        n_times = da.sizes.get("time", 0)
        logger.info(
            "Rainfall stored: %d timestep(s), grid %d × %d.",
            n_times,
            da.sizes.get("y", 0),
            da.sizes.get("x", 0),
        )
        return da

    def compute_layers(
        self,
        dem_path: Union[str, Path],
        *,
        rainfall_start: Union[str, datetime, None] = None,
        rainfall_end: Union[str, datetime, None] = None,
        rainfall_nimrod_dir: Optional[Union[str, Path]] = None,
        topo_res: int = 50,
        slope_output: Literal["degrees", "percent_rise", "radians"] = "degrees",
        aspect_output: Literal["degrees", "radians"] = "degrees",
        soil_timeout: int = 60,
        delineate_if_missing: bool = True,
        flow_acc_threshold: int = 5000,
    ) -> CatchmentLayers:
        """Compute all spatial layers for this catchment in a single call.

        Convenience wrapper that calls (in order):

        1. :meth:`delineate` — only when *delineate_if_missing* is ``True``
           and :attr:`has_catchment` is ``False``.
        2. :meth:`compute_topography`
        3. :meth:`compute_slope`
        4. :meth:`compute_aspect`
        5. :meth:`compute_soil_type`
        6. :meth:`compute_rainfall` — only when *rainfall_start*,
           *rainfall_end*, and *rainfall_nimrod_dir* are all provided.

        Each step is attempted independently: a failure in one step is logged
        as a warning and execution continues with the next step rather than
        aborting the whole run.

        Parameters
        ----------
        dem_path : str or Path
            Path to a GeoTIFF DEM used for delineation, topography, slope,
            and aspect computation.
        rainfall_start, rainfall_end : str or datetime, optional
            Time window for rainfall extraction.  If either is ``None``
            (the default) the rainfall step is skipped.
        rainfall_nimrod_dir : str or Path, optional
            Path to the NIMROD data directory (or consolidated ``.nc`` file)
            used by :meth:`compute_rainfall`.  Required when
            *rainfall_start* and *rainfall_end* are provided.
        topo_res : int
            BNG resolution for the topography, slope, and aspect grids in
            metres (default: 50 m).
        slope_output : str
            Slope units — ``"degrees"`` (default), ``"percent_rise"``, or
            ``"radians"``.
        aspect_output : str
            Aspect units — ``"degrees"`` (default, range 0–360 clockwise from
            North) or ``"radians"``.
        soil_timeout : int
            HTTP timeout in seconds for the SEARG soil-data request
            (default: 60 s).
        delineate_if_missing : bool
            If ``True`` (default) and this sample has no catchment polygon,
            attempt to delineate one from *dem_path* before computing layers.
        flow_acc_threshold : int
            Passed to :meth:`delineate` when delineation is triggered.

        Returns
        -------
        CatchmentLayers
            The populated :attr:`layers` object (same reference as
            ``self.layers``).
        """
        # --- optional delineation -----------------------------------------
        if delineate_if_missing and not self.has_catchment:
            logger.info(
                "No catchment polygon for site '%s' — attempting delineation …",
                self.site_name,
            )
            try:
                self.delineate(dem_path, flow_acc_threshold=flow_acc_threshold)
            except Exception as exc:
                logger.warning(
                    "Delineation failed for site '%s': %s — skipping all layers.",
                    self.site_name,
                    exc,
                )
                return self.layers

        if not self.has_catchment:
            logger.warning(
                "Site '%s' has no catchment polygon — skipping layer computation.",
                self.site_name,
            )
            return self.layers

        # --- topography ---------------------------------------------------
        try:
            self.compute_topography(dem_path, res=topo_res)
        except Exception as exc:
            logger.warning(
                "Topography computation failed for site '%s': %s",
                self.site_name,
                exc,
            )

        # --- slope --------------------------------------------------------
        try:
            self.compute_slope(dem_path, res=topo_res, output=slope_output)
        except Exception as exc:
            logger.warning(
                "Slope computation failed for site '%s': %s",
                self.site_name,
                exc,
            )

        # --- aspect -------------------------------------------------------
        try:
            self.compute_aspect(dem_path, res=topo_res, output=aspect_output)
        except Exception as exc:
            logger.warning(
                "Aspect computation failed for site '%s': %s",
                self.site_name,
                exc,
            )

        # --- soil type ----------------------------------------------------
        try:
            self.compute_soil_type(timeout=soil_timeout)
        except Exception as exc:
            logger.warning(
                "Soil type query failed for site '%s': %s",
                self.site_name,
                exc,
            )

        # --- rainfall (optional) ------------------------------------------
        if (
            rainfall_start is not None
            and rainfall_end is not None
            and rainfall_nimrod_dir is not None
        ):
            try:
                self.compute_rainfall(
                    rainfall_start,
                    rainfall_end,
                    nimrod_dir=rainfall_nimrod_dir,
                )
            except Exception as exc:
                logger.warning(
                    "Rainfall computation failed for site '%s': %s",
                    self.site_name,
                    exc,
                )
        else:
            logger.debug(
                "Rainfall skipped for site '%s' "
                "(rainfall_start, rainfall_end, or rainfall_nimrod_dir not provided).",
                self.site_name,
            )

        logger.info(
            "compute_layers complete for site '%s': layers=%s",
            self.site_name,
            self.layers.available,
        )
        return self.layers

    # ------------------------------------------------------------------
    # openEO connection helper
    # ------------------------------------------------------------------

    @staticmethod
    def connect_openeo(
        backend: str = OPENEO_BACKEND,
        *,
        authenticate: bool = True,
    ) -> "openeo.Connection":
        """Connect to an openEO backend and (optionally) authenticate.

        Parameters
        ----------
        backend : str
            URL of the openEO backend.  Defaults to the Copernicus Data
            Space Federation endpoint.
        authenticate : bool
            If *True* (default), trigger OIDC device-flow authentication.
            Pass *False* to get an unauthenticated connection (only publicly
            accessible collections will be accessible).

        Returns
        -------
        openeo.Connection

        Raises
        ------
        ImportError
            If the ``openeo`` package is not installed.
        """
        from eoflow import eo as _eo

        return _eo.connect(backend, authenticate=authenticate)

    # ------------------------------------------------------------------
    # Core EO query
    # ------------------------------------------------------------------

    def query_bands(
        self,
        connection: "openeo.Connection",
        bands: Sequence[str],
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
        *,
        days_window: Optional[int] = None,
        collection: str = SENTINEL2_COLLECTION,
        max_cloud_cover: int = 85,
    ) -> "openeo.DataCube":
        """Load a Sentinel-2 data cube for the catchment polygon.

        Parameters
        ----------
        connection : openeo.Connection
            An authenticated openEO connection (see :meth:`connect_openeo`).
        bands : sequence of str
            Band names to load, e.g. ``["B04", "B08"]``.
        start_date, end_date : str or date, optional
            Temporal extent as ISO-8601 strings (``"YYYY-MM-DD"``) or
            :class:`datetime.date` objects.  If both are *None* and
            *days_window* is also *None*, defaults to a 30-day window
            centred on the sample date (or raises :exc:`ValueError` if the
            sample has no date).
        days_window : int, optional
            If provided, *start_date* and *end_date* are computed
            automatically as ``sample.date ± days_window`` days.  Overrides
            explicit *start_date*/*end_date* values.
        collection : str
            openEO collection ID.  Defaults to ``SENTINEL2_L2A``.
        max_cloud_cover : int
            Maximum cloud cover percentage (0–100).  Passed to
            ``load_collection`` to pre-filter cloudy scenes.

        Returns
        -------
        openeo.DataCube
            The resulting data cube.  Call ``.download()`` or
            ``.execute_batch()`` to materialise it.

        Raises
        ------
        ValueError
            If the sample has no catchment polygon, or if dates cannot be
            determined.
        """
        if not self.has_catchment:
            raise ValueError(
                f"Sample '{self.id}' has no delineated catchment polygon — "
                "cannot define a spatial extent for the EO query."
            )

        start_str, end_str = self._resolve_dates(start_date, end_date, days_window)
        spatial_extent = self._spatial_extent_dict()

        from eoflow import eo as _eo

        logger.info(
            "Building band cube for site '%s' | bands=%s | %s → %s",
            self.site_name,
            list(bands),
            start_str,
            end_str,
        )
        return _eo.build_band_cube(
            connection,
            spatial_extent,
            (start_str, end_str),
            bands,
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )

    # ------------------------------------------------------------------
    # Named spectral index helpers
    # ------------------------------------------------------------------

    def query_ndvi(
        self,
        connection: "openeo.Connection",
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
        *,
        days_window: Optional[int] = None,
        nir_band: str = "B08",
        red_band: str = "B04",
        collection: str = SENTINEL2_COLLECTION,
        max_cloud_cover: int = 85,
    ) -> "openeo.DataCube":
        """Query NDVI = (NIR − Red) / (NIR + Red) for the catchment.

        Parameters
        ----------
        connection : openeo.Connection
        start_date, end_date : str or date, optional
        days_window : int, optional
            See :meth:`query_bands` for documentation of these parameters.
        nir_band : str
            Name of the NIR band.  Default ``"B08"`` (Sentinel-2 10 m).
        red_band : str
            Name of the red band.  Default ``"B04"``.
        collection : str
        max_cloud_cover : int

        Returns
        -------
        openeo.DataCube
            Single-band cube containing NDVI values (−1 to 1).
        """
        cube = self.query_bands(
            connection,
            bands=[nir_band, red_band],
            start_date=start_date,
            end_date=end_date,
            days_window=days_window,
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        ndvi = cube.ndvi(nir=nir_band, red=red_band)
        logger.info("NDVI process applied (nir=%s, red=%s)", nir_band, red_band)
        return ndvi

    def query_ndwi(
        self,
        connection: "openeo.Connection",
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
        *,
        days_window: Optional[int] = None,
        green_band: str = "B03",
        nir_band: str = "B08",
        collection: str = SENTINEL2_COLLECTION,
        max_cloud_cover: int = 85,
    ) -> "openeo.DataCube":
        """Query NDWI = (Green − NIR) / (Green + NIR) for the catchment.

        NDWI highlights open water surfaces (McFeeters 1996).

        Parameters
        ----------
        connection : openeo.Connection
        start_date, end_date : str or date, optional
        days_window : int, optional
        green_band : str
            Default ``"B03"`` (Sentinel-2 green, 10 m).
        nir_band : str
            Default ``"B08"`` (Sentinel-2 NIR, 10 m).
        collection : str
        max_cloud_cover : int

        Returns
        -------
        openeo.DataCube
            Single-band cube containing NDWI values (−1 to 1).
        """
        cube = self.query_bands(
            connection,
            bands=[green_band, nir_band],
            start_date=start_date,
            end_date=end_date,
            days_window=days_window,
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        from eoflow import eo as _eo

        return _eo.compute_ndwi(cube, green_band=green_band, nir_band=nir_band)

    def query_index(
        self,
        connection: "openeo.Connection",
        index: str,
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
        *,
        days_window: Optional[int] = None,
        collection: str = SENTINEL2_COLLECTION,
        max_cloud_cover: int = 85,
    ) -> "openeo.DataCube":
        """Query a named spectral index for the catchment.

        Supported index names (case-insensitive):

        * ``"NDVI"``  — Normalised Difference Vegetation Index
        * ``"NDWI"``  — Normalised Difference Water Index (McFeeters 1996)
        * ``"MNDWI"`` — Modified NDWI using SWIR (Xu 2006) — ``(B03 − B11) / (B03 + B11)``
        * ``"NDRE"``  — Normalised Difference Red Edge ``(B05 − B04) / (B05 + B04)``
        * ``"EVI"``   — Enhanced Vegetation Index (3-band; approximated as
          ``2.5 * (NIR − Red) / (NIR + 6*Red − 7.5*Blue + 1)``)

        Parameters
        ----------
        connection : openeo.Connection
        index : str
            Name of the index (see above).
        start_date, end_date : str or date, optional
        days_window : int, optional
        collection : str
        max_cloud_cover : int

        Returns
        -------
        openeo.DataCube

        Raises
        ------
        ValueError
            If *index* is not one of the supported names.
        """
        index_upper = index.upper()
        if index_upper == "NDVI":
            return self.query_ndvi(
                connection,
                start_date=start_date,
                end_date=end_date,
                days_window=days_window,
                collection=collection,
                max_cloud_cover=max_cloud_cover,
            )
        if index_upper == "NDWI":
            return self.query_ndwi(
                connection,
                start_date=start_date,
                end_date=end_date,
                days_window=days_window,
                collection=collection,
                max_cloud_cover=max_cloud_cover,
            )
        if index_upper == "MNDWI":
            return self._query_normalised_difference(
                connection,
                band_a="B03",
                band_b="B11",
                label="MNDWI",
                start_date=start_date,
                end_date=end_date,
                days_window=days_window,
                collection=collection,
                max_cloud_cover=max_cloud_cover,
            )
        if index_upper == "NDRE":
            return self._query_normalised_difference(
                connection,
                band_a="B05",
                band_b="B04",
                label="NDRE",
                start_date=start_date,
                end_date=end_date,
                days_window=days_window,
                collection=collection,
                max_cloud_cover=max_cloud_cover,
            )
        if index_upper == "EVI":
            return self._query_evi(
                connection,
                start_date=start_date,
                end_date=end_date,
                days_window=days_window,
                collection=collection,
                max_cloud_cover=max_cloud_cover,
            )

        supported = ", ".join(sorted(_S2_BANDS.keys()))
        raise ValueError(
            f"Unknown index '{index}'. Supported indices: {supported}. "
            "For arbitrary bands use query_bands() instead."
        )

    # ------------------------------------------------------------------
    # Private EO helpers
    # ------------------------------------------------------------------

    def _query_normalised_difference(
        self,
        connection: "openeo.Connection",
        band_a: str,
        band_b: str,
        label: str,
        start_date: Union[str, date, None],
        end_date: Union[str, date, None],
        days_window: Optional[int],
        collection: str,
        max_cloud_cover: int,
    ) -> "openeo.DataCube":
        """Generic (band_a − band_b) / (band_a + band_b) normalised difference."""
        cube = self.query_bands(
            connection,
            bands=[band_a, band_b],
            start_date=start_date,
            end_date=end_date,
            days_window=days_window,
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        from eoflow import eo as _eo

        return _eo.compute_normalised_difference(cube, band_a, band_b, label)

    def _query_evi(
        self,
        connection: "openeo.Connection",
        start_date: Union[str, date, None],
        end_date: Union[str, date, None],
        days_window: Optional[int],
        collection: str,
        max_cloud_cover: int,
    ) -> "openeo.DataCube":
        """EVI = 2.5 * (NIR − Red) / (NIR + 6*Red − 7.5*Blue + 1)."""
        from eoflow import eo as _eo

        nir_b, red_b, blue_b = "B08", "B04", "B02"
        cube = self.query_bands(
            connection,
            bands=[nir_b, red_b, blue_b],
            start_date=start_date,
            end_date=end_date,
            days_window=days_window,
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        return _eo.compute_evi(cube, nir_band=nir_b, red_band=red_b, blue_band=blue_b)

    # ------------------------------------------------------------------
    # Materialising EO queries (download + store on self.layers)
    # ------------------------------------------------------------------

    def fetch_bands(
        self,
        connection: "openeo.Connection",
        bands: Sequence[str],
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
        *,
        days_window: Optional[int] = None,
        collection: str = SENTINEL2_COLLECTION,
        max_cloud_cover: int = 85,
    ) -> "xarray.Dataset":
        """Download Sentinel-2 band data for the catchment and store it.

        Builds the openEO datacube via :meth:`query_bands`, executes it
        synchronously, and stores the resulting
        :class:`xarray.Dataset` in :attr:`layers.eo_bands`.

        Parameters
        ----------
        connection : openeo.Connection
            An authenticated openEO connection (see :meth:`connect_openeo`).
        bands : sequence of str
            Band names to download, e.g. ``["B04", "B08"]``.
        start_date, end_date : str or date, optional
            Temporal extent as ISO-8601 strings or :class:`datetime.date`
            objects.  If both are *None* and *days_window* is also *None*,
            defaults to a 30-day window around the sample date.
        days_window : int, optional
            Compute dates symmetrically as ``sample.date ± days_window`` days.
        collection : str
            openEO collection ID.  Defaults to ``SENTINEL2_L2A``.
        max_cloud_cover : int
            Maximum cloud cover percentage (0–100).

        Returns
        -------
        xarray.Dataset
            The downloaded dataset, also stored as ``self.layers.eo_bands``.

        Raises
        ------
        ValueError
            If the sample has no catchment polygon, or if dates cannot be
            determined.
        """
        from eoflow import eo as _eo

        cube = self.query_bands(
            connection,
            bands=bands,
            start_date=start_date,
            end_date=end_date,
            days_window=days_window,
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        logger.info(
            "Downloading band data for site '%s' (bands=%s) …",
            self.site_name,
            list(bands),
        )
        ds = _eo.download_cube_as_xarray(cube)
        self.layers.eo_bands = ds
        logger.info(
            "EO bands stored for site '%s': vars=%s",
            self.site_name,
            list(ds.data_vars),
        )
        return ds

    def fetch_ndvi(
        self,
        connection: "openeo.Connection",
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
        *,
        days_window: Optional[int] = None,
        nir_band: str = "B08",
        red_band: str = "B04",
        collection: str = SENTINEL2_COLLECTION,
        max_cloud_cover: int = 85,
    ) -> "xarray.DataArray":
        """Download NDVI for the catchment and store it in ``self.layers.ndvi``.

        Builds the NDVI datacube via :meth:`query_ndvi`, executes it
        synchronously, and stores the single-band result as a
        :class:`xarray.DataArray` with dimensions ``(t, y, x)`` (exact
        dimension names depend on the backend).

        Parameters
        ----------
        connection : openeo.Connection
            An authenticated openEO connection (see :meth:`connect_openeo`).
        start_date, end_date : str or date, optional
            Temporal extent.  Defaults to a 30-day window around the sample
            date when both are *None*.
        days_window : int, optional
            Compute dates symmetrically as ``sample.date ± days_window`` days.
        nir_band : str
            NIR band name.  Default ``"B08"`` (Sentinel-2 10 m).
        red_band : str
            Red band name.  Default ``"B04"`` (Sentinel-2 10 m).
        collection : str
        max_cloud_cover : int

        Returns
        -------
        xarray.DataArray
            NDVI values in [−1, 1], also stored as ``self.layers.ndvi``.

        Raises
        ------
        ValueError
            If the sample has no catchment polygon, or if dates cannot be
            determined.

        Examples
        --------
        ::

            conn = Sample.connect_openeo()
            ndvi = sample.fetch_ndvi(conn, start_date="2024-03-01", end_date="2024-04-30")
            ndvi.mean().item()   # mean NDVI over the catchment/period
        """
        from eoflow import eo as _eo

        cube = self.query_ndvi(
            connection,
            start_date=start_date,
            end_date=end_date,
            days_window=days_window,
            nir_band=nir_band,
            red_band=red_band,
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        logger.info("Downloading NDVI for site '%s' …", self.site_name)
        ds = _eo.download_cube_as_xarray(cube)
        da = _eo.extract_dataarray(ds, name="ndvi")
        self.layers.ndvi = da
        logger.info(
            "NDVI stored for site '%s': dims=%s",
            self.site_name,
            da.dims,
        )
        return da

    def fetch_ndwi(
        self,
        connection: "openeo.Connection",
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
        *,
        days_window: Optional[int] = None,
        green_band: str = "B03",
        nir_band: str = "B08",
        collection: str = SENTINEL2_COLLECTION,
        max_cloud_cover: int = 85,
    ) -> "xarray.DataArray":
        """Download NDWI for the catchment and store it in ``self.layers.ndwi``.

        Builds the NDWI datacube via :meth:`query_ndwi`, executes it
        synchronously, and stores the single-band result as a
        :class:`xarray.DataArray`.

        Parameters
        ----------
        connection : openeo.Connection
            An authenticated openEO connection (see :meth:`connect_openeo`).
        start_date, end_date : str or date, optional
            Temporal extent.  Defaults to a 30-day window around the sample
            date when both are *None*.
        days_window : int, optional
            Compute dates symmetrically as ``sample.date ± days_window`` days.
        green_band : str
            Green band name.  Default ``"B03"`` (Sentinel-2 10 m).
        nir_band : str
            NIR band name.  Default ``"B08"`` (Sentinel-2 10 m).
        collection : str
        max_cloud_cover : int

        Returns
        -------
        xarray.DataArray
            NDWI values in [−1, 1], also stored as ``self.layers.ndwi``.

        Raises
        ------
        ValueError
            If the sample has no catchment polygon, or if dates cannot be
            determined.

        Examples
        --------
        ::

            conn = Sample.connect_openeo()
            ndwi = sample.fetch_ndwi(conn, start_date="2024-03-01", end_date="2024-04-30")
        """
        from eoflow import eo as _eo

        cube = self.query_ndwi(
            connection,
            start_date=start_date,
            end_date=end_date,
            days_window=days_window,
            green_band=green_band,
            nir_band=nir_band,
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        logger.info("Downloading NDWI for site '%s' …", self.site_name)
        ds = _eo.download_cube_as_xarray(cube)
        da = _eo.extract_dataarray(ds, name="ndwi")
        self.layers.ndwi = da
        logger.info(
            "NDWI stored for site '%s': dims=%s",
            self.site_name,
            da.dims,
        )
        return da

    def fetch_index(
        self,
        connection: "openeo.Connection",
        index: str,
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
        *,
        days_window: Optional[int] = None,
        collection: str = SENTINEL2_COLLECTION,
        max_cloud_cover: int = 85,
    ) -> "xarray.DataArray":
        """Download a named spectral index and store it on ``self.layers``.

        This is a dispatch wrapper around :meth:`fetch_ndvi`,
        :meth:`fetch_ndwi`, and the lower-level :func:`eoflow.eo.build_index_cube`
        + :func:`eoflow.eo.download_cube_as_xarray` pipeline for other indices.

        Results for NDVI and NDWI are stored in the dedicated
        ``self.layers.ndvi`` / ``self.layers.ndwi`` slots.  For all other
        supported indices (MNDWI, NDRE, EVI) the DataArray is returned but
        not automatically persisted to a named slot — callers may assign it
        manually (e.g. ``sample.layers.eo_bands = ...``).

        Supported index names (case-insensitive):
        ``NDVI``, ``NDWI``, ``MNDWI``, ``NDRE``, ``EVI``.

        Parameters
        ----------
        connection : openeo.Connection
            An authenticated openEO connection (see :meth:`connect_openeo`).
        index : str
            Spectral index name (see above).
        start_date, end_date : str or date, optional
            Temporal extent.  Defaults to a 30-day window around the sample
            date when both are *None*.
        days_window : int, optional
            Compute dates symmetrically as ``sample.date ± days_window`` days.
        collection : str
        max_cloud_cover : int

        Returns
        -------
        xarray.DataArray
            Index values, also stored on ``self.layers`` for NDVI/NDWI.

        Raises
        ------
        ValueError
            If *index* is not one of the supported names, the sample has no
            catchment polygon, or dates cannot be determined.

        Examples
        --------
        ::

            conn = Sample.connect_openeo()
            ndvi = sample.fetch_index(conn, "NDVI", days_window=30)
            ndwi = sample.fetch_index(conn, "NDWI", days_window=30)
        """
        idx = index.upper()

        if idx == "NDVI":
            return self.fetch_ndvi(
                connection,
                start_date=start_date,
                end_date=end_date,
                days_window=days_window,
                collection=collection,
                max_cloud_cover=max_cloud_cover,
            )

        if idx == "NDWI":
            return self.fetch_ndwi(
                connection,
                start_date=start_date,
                end_date=end_date,
                days_window=days_window,
                collection=collection,
                max_cloud_cover=max_cloud_cover,
            )

        # For MNDWI / NDRE / EVI — build and download via eo module
        from eoflow import eo as _eo

        cube = self.query_index(
            connection,
            index=index,
            start_date=start_date,
            end_date=end_date,
            days_window=days_window,
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        logger.info("Downloading %s for site '%s' …", idx, self.site_name)
        ds = _eo.download_cube_as_xarray(cube)
        da = _eo.extract_dataarray(ds, name=idx.lower())
        logger.info(
            "%s downloaded for site '%s': dims=%s",
            idx,
            self.site_name,
            da.dims,
        )
        return da

    def _resolve_dates(
        self,
        start_date: Union[str, date, None],
        end_date: Union[str, date, None],
        days_window: Optional[int],
    ) -> Tuple[str, str]:
        """Return (start_str, end_str) ISO-8601 strings for the query window.

        Priority:
        1. *days_window* — compute symmetrically around ``self.date``.
        2. Explicit *start_date* / *end_date*.
        3. Default: 30-day window around ``self.date``.
        """
        if days_window is not None:
            centre = self.date
            if centre is None:
                raise ValueError(
                    "Cannot use days_window: sample has no date. "
                    "Provide explicit start_date and end_date instead."
                )
            delta = timedelta(days=days_window)
            return (centre - delta).isoformat(), (centre + delta).isoformat()

        if start_date is not None and end_date is not None:
            return _to_iso(start_date), _to_iso(end_date)

        if start_date is not None or end_date is not None:
            warnings.warn(
                "Only one of start_date / end_date was supplied; "
                "defaulting to a 30-day window around the sample date.",
                stacklevel=3,
            )

        centre = self.date
        if centre is None:
            raise ValueError(
                "Sample has no date and no date range was provided. "
                "Pass start_date and end_date explicitly."
            )
        delta = timedelta(days=30)
        return (centre - delta).isoformat(), (centre + delta).isoformat()

    def _spatial_extent_dict(self) -> Dict[str, float]:
        """Return the openEO spatial_extent dict (west/south/east/north)."""
        bbox = self.catchment_bbox()
        if bbox is None:
            raise ValueError("No catchment polygon available to define spatial extent.")
        west, south, east, north = bbox
        return {"west": west, "south": south, "east": east, "north": north}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save this Sample to a directory on disk.

        The directory will contain the following files:

        - ``metadata.json`` — the observation row serialised as JSON.
        - ``catchment.wkt`` — the catchment polygon as WKT (omitted when
          no catchment is present).
        - ``layers/topography.nc`` — topography :class:`xarray.DataArray`
          as NetCDF (omitted when not yet computed).
        - ``layers/slope.nc`` — slope :class:`xarray.DataArray` as NetCDF
          (omitted when not yet computed).
        - ``layers/soil_type.json`` — soil-type :class:`pandas.Series` as
          JSON (omitted when not yet computed).
        - ``layers/rainfall.nc`` — rainfall :class:`xarray.DataArray` as
          NetCDF (omitted when not yet computed).

        Parameters
        ----------
        path : str or Path
            Directory to write into.  Created (including any missing
            parents) if it does not already exist.
        """
        from shapely import wkt as shapely_wkt

        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        # 1. Observation-row metadata
        (root / "metadata.json").write_text(self._row.to_json() or "", encoding="utf-8")
        logger.debug("Saved metadata to %s", root / "metadata.json")

        # 2. Catchment geometry
        if self.has_catchment:
            geom = self.catchment
            assert geom is not None  # guaranteed by has_catchment check above
            wkt_text = shapely_wkt.dumps(geom)
            (root / "catchment.wkt").write_text(wkt_text, encoding="utf-8")
            logger.debug("Saved catchment geometry to %s", root / "catchment.wkt")

        # 3. Computed spatial layers
        layers_dir = root / "layers"

        if self.layers.topography is not None:
            layers_dir.mkdir(parents=True, exist_ok=True)
            da = self.layers.topography
            if da.name is None:
                da = da.rename("topography")
            da.to_netcdf(layers_dir / "topography.nc")
            logger.debug("Saved topography layer to %s", layers_dir / "topography.nc")

        if self.layers.slope is not None:
            layers_dir.mkdir(parents=True, exist_ok=True)
            da = self.layers.slope
            if da.name is None:
                da = da.rename("slope")
            da.to_netcdf(layers_dir / "slope.nc")
            logger.debug("Saved slope layer to %s", layers_dir / "slope.nc")

        if self.layers.aspect is not None:
            layers_dir.mkdir(parents=True, exist_ok=True)
            da = self.layers.aspect
            if da.name is None:
                da = da.rename("aspect")
            da.to_netcdf(layers_dir / "aspect.nc")
            logger.debug("Saved aspect layer to %s", layers_dir / "aspect.nc")

        if self.layers.soil_type is not None:
            layers_dir.mkdir(parents=True, exist_ok=True)
            (layers_dir / "soil_type.json").write_text(
                self.layers.soil_type.to_json() or "", encoding="utf-8"
            )
            logger.debug("Saved soil_type layer to %s", layers_dir / "soil_type.json")

        if self.layers.soil_polygons is not None:
            layers_dir.mkdir(parents=True, exist_ok=True)
            self.layers.soil_polygons.to_file(
                layers_dir / "soil_polygons.geojson", driver="GeoJSON"
            )
            logger.debug("Saved soil polygons to %s", layers_dir / "soil_polygons.geojson")

        if self.layers.rainfall is not None:
            layers_dir.mkdir(parents=True, exist_ok=True)
            da = self.layers.rainfall
            if da.name is None:
                da = da.rename("rainfall")
            da.to_netcdf(layers_dir / "rainfall.nc")
            logger.debug("Saved rainfall layer to %s", layers_dir / "rainfall.nc")

        if self.layers.ndvi is not None and self.layers.ndvi.size > 0:
            layers_dir.mkdir(parents=True, exist_ok=True)
            da = self.layers.ndvi
            if da.name is None:
                da = da.rename("ndvi")
            da.to_netcdf(layers_dir / "ndvi.nc")
            logger.debug("Saved NDVI layer to %s", layers_dir / "ndvi.nc")
        elif self.layers.ndvi is not None:
            logger.warning("Skipping save of empty NDVI DataArray (size=0)")

        if self.layers.ndwi is not None and self.layers.ndwi.size > 0:
            layers_dir.mkdir(parents=True, exist_ok=True)
            da = self.layers.ndwi
            if da.name is None:
                da = da.rename("ndwi")
            da.to_netcdf(layers_dir / "ndwi.nc")
            logger.debug("Saved NDWI layer to %s", layers_dir / "ndwi.nc")
        elif self.layers.ndwi is not None:
            logger.warning("Skipping save of empty NDWI DataArray (size=0)")

        if self.layers.eo_bands is not None and len(self.layers.eo_bands.data_vars) > 0:
            layers_dir.mkdir(parents=True, exist_ok=True)
            self.layers.eo_bands.to_netcdf(layers_dir / "eo_bands.nc")
            logger.debug("Saved EO bands to %s", layers_dir / "eo_bands.nc")
        elif self.layers.eo_bands is not None:
            logger.warning("Skipping save of empty EO bands Dataset (no data variables)")

        logger.info("Sample '%s' saved to %s", self.id, root)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Sample":
        """Load a :class:`Sample` that was previously saved with :meth:`save`.

        Parameters
        ----------
        path : str or Path
            Directory written by :meth:`save`.

        Returns
        -------
        Sample
            A fully reconstructed instance, including any spatial layers
            that were present when :meth:`save` was called.

        Raises
        ------
        FileNotFoundError
            If *path* is not a directory or ``metadata.json`` is missing.
        """
        import xarray
        from shapely import wkt as shapely_wkt

        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"No such directory: {root}")

        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {root}. Is this a valid Sample directory?"
            )

        # 1. Observation-row metadata
        row = pd.read_json(
            metadata_path,
            typ="series",
            dtype=False,  # type: ignore[arg-type]
            convert_dates=False,
        )
        logger.debug("Loaded metadata from %s", metadata_path)

        # 2. Catchment geometry
        catchment: Optional[BaseGeometry] = None
        wkt_path = root / "catchment.wkt"
        if wkt_path.exists():
            catchment = shapely_wkt.loads(wkt_path.read_text(encoding="utf-8"))
            logger.debug("Loaded catchment geometry from %s", wkt_path)

        sample = cls(row, catchment)

        # 3. Computed spatial layers
        layers_dir = root / "layers"
        if layers_dir.is_dir():
            topo_path = layers_dir / "topography.nc"
            if topo_path.exists():
                sample.layers.topography = xarray.open_dataarray(topo_path).load()
                logger.debug("Loaded topography layer from %s", topo_path)

            slope_path = layers_dir / "slope.nc"
            if slope_path.exists():
                sample.layers.slope = xarray.open_dataarray(slope_path).load()
                logger.debug("Loaded slope layer from %s", slope_path)

            aspect_path = layers_dir / "aspect.nc"
            if aspect_path.exists():
                sample.layers.aspect = xarray.open_dataarray(aspect_path).load()
                logger.debug("Loaded aspect layer from %s", aspect_path)

            soil_path = layers_dir / "soil_type.json"
            if soil_path.exists():
                sample.layers.soil_type = pd.read_json(
                    soil_path,
                    typ="series",
                    dtype=False,  # type: ignore[arg-type]
                )
                logger.debug("Loaded soil_type layer from %s", soil_path)

            soil_polygons_path = layers_dir / "soil_polygons.geojson"
            if soil_polygons_path.exists():
                sample.layers.soil_polygons = gpd.read_file(soil_polygons_path)
                logger.debug("Loaded soil polygons from %s", soil_polygons_path)

            rainfall_path = layers_dir / "rainfall.nc"
            if rainfall_path.exists():
                sample.layers.rainfall = xarray.open_dataarray(rainfall_path).load()
                logger.debug("Loaded rainfall layer from %s", rainfall_path)

            ndvi_path = layers_dir / "ndvi.nc"
            if ndvi_path.exists():
                try:
                    sample.layers.ndvi = xarray.open_dataarray(ndvi_path).load()
                    logger.debug("Loaded NDVI layer from %s", ndvi_path)
                except ValueError as exc:
                    logger.warning("Skipping empty/corrupt NDVI file %s: %s", ndvi_path, exc)

            ndwi_path = layers_dir / "ndwi.nc"
            if ndwi_path.exists():
                try:
                    sample.layers.ndwi = xarray.open_dataarray(ndwi_path).load()
                    logger.debug("Loaded NDWI layer from %s", ndwi_path)
                except ValueError as exc:
                    logger.warning("Skipping empty/corrupt NDWI file %s: %s", ndwi_path, exc)

            eo_bands_path = layers_dir / "eo_bands.nc"
            if eo_bands_path.exists():
                try:
                    sample.layers.eo_bands = xarray.open_dataset(eo_bands_path).load()
                    logger.debug("Loaded EO bands from %s", eo_bands_path)
                except ValueError as exc:
                    logger.warning(
                        "Skipping empty/corrupt EO bands file %s: %s", eo_bands_path, exc
                    )

        logger.info("Sample '%s' loaded from %s", sample.id, root)
        return sample

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Sample("
            f"id={self.id!r}, "
            f"site={self.site_name!r}, "
            f"date={self.date!r}, "
            f"result={self.result!r}, "
            f"has_catchment={self.has_catchment}"
            f")"
        )

    def __str__(self) -> str:
        area = self.catchment_area_km2()
        area_str = f"{area:.3f} km²" if area is not None else "n/a"
        return (
            f"Sample\n"
            f"  Site        : {self.site_name}\n"
            f"  Notation    : {self.notation}\n"
            f"  Date        : {self.date}\n"
            f"  Measurement : {self.result} {self.unit} ({self.determinand})\n"
            f"  Location    : ({self.latitude:.5f}, {self.longitude:.5f})\n"
            f"  Catchment   : {self.delineation_status} — area ≈ {area_str}\n"
            f"  Flow acc    : {self.flow_acc_at_pour_point} cells"
        )


#: Deprecated backward-compatibility alias.  :class:`Sample` is the
#: canonical name.  ``CatchmentSample`` will be removed in a future
#: release — update any existing code to use :class:`Sample` directly.
CatchmentSample = Sample

# ---------------------------------------------------------------------------
# CatchmentDataset
# ---------------------------------------------------------------------------


class CatchmentDataset:
    """A collection of :class:`Sample` objects.

    Load from the GeoPackage produced by ``delineate_catchments.py`` with
    :meth:`from_gpkg`, then iterate, filter, and query EO data.

    Parameters
    ----------
    samples : list of Sample
        The samples in the dataset.
    source_path : Path or None
        Path to the originating GeoPackage (informational).
    """

    def __init__(
        self,
        samples: List[Sample],
        source_path: Optional[Path] = None,
    ) -> None:
        self._samples: List[Sample] = samples
        self.source_path: Optional[Path] = source_path

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_gpkg(
        cls,
        gpkg_path: Union[str, Path],
        layer: Optional[str] = None,
        *,
        only_delineated: bool = False,
    ) -> "CatchmentDataset":
        """Load a :class:`CatchmentDataset` from a GeoPackage file.

        The GeoPackage must be in the format written by ``delineate_catchments.py``
        — specifically it needs a geometry column holding catchment polygons
        and (optionally) ``latitude``, ``longitude``, ``snap_latitude``,
        ``snap_longitude``, ``flow_acc_at_pour_point`` columns.

        Parameters
        ----------
        gpkg_path : str or Path
            Path to the GeoPackage (``.gpkg``) file.
        layer : str, optional
            Layer name within the GeoPackage.  When *None* (default) the
            first layer is used.
        only_delineated : bool
            If *True*, drop rows where ``delineation_status != "ok"``.

        Returns
        -------
        CatchmentDataset

        Raises
        ------
        FileNotFoundError
            If *gpkg_path* does not exist.
        ValueError
            If the GeoPackage contains no rows.
        """
        gpkg_path = Path(gpkg_path)
        if not gpkg_path.exists():
            raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

        kwargs: Dict[str, Any] = {}
        if layer is not None:
            kwargs["layer"] = layer

        gdf = gpd.read_file(str(gpkg_path), **kwargs)
        logger.info("Loaded %d rows from %s", len(gdf), gpkg_path)

        if gdf.empty:
            raise ValueError(f"GeoPackage '{gpkg_path}' contains no rows.")

        # Ensure WGS 84
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
            logger.debug("Reprojected GeoDataFrame to EPSG:4326")

        if only_delineated and _COL_STATUS in gdf.columns:
            before = len(gdf)
            gdf = gdf[gdf[_COL_STATUS] == "ok"].copy()
            logger.info("only_delineated=True: kept %d / %d rows", len(gdf), before)

        samples: List[Sample] = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is not None and not (isinstance(geom, float) and math.isnan(geom)):
                try:
                    catchment = geom if isinstance(geom, Polygon) else None
                except Exception:
                    catchment = None
            else:
                catchment = None

            # If __catchment_wkt is present and geometry column is not a
            # polygon, try recovering from WKT
            if catchment is None and _COL_WKT in row.index:
                wkt = row[_COL_WKT]
                if isinstance(wkt, str) and wkt.startswith("POLYGON"):
                    try:
                        from shapely import wkt as shapely_wkt

                        catchment = shapely_wkt.loads(wkt)
                    except Exception:
                        pass

            samples.append(Sample(row=row, catchment=catchment))  # type: ignore[arg-type]

        n_with = sum(1 for s in samples if s.has_catchment)
        logger.info(
            "CatchmentDataset built: %d samples, %d with catchments",
            len(samples),
            n_with,
        )
        return cls(samples, source_path=gpkg_path)

    @classmethod
    def from_geodataframe(
        cls,
        gdf: gpd.GeoDataFrame,
        source_path: Optional[Path] = None,
    ) -> "CatchmentDataset":
        """Construct a :class:`CatchmentDataset` directly from a GeoDataFrame.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame in the format produced by ``delineate_catchments.py``.
        source_path : Path, optional
            Optional path to record as the data source.
        """
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        samples = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            catchment = geom if isinstance(geom, Polygon) else None
            samples.append(Sample(row=row, catchment=catchment))

        return cls(samples, source_path=source_path)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def with_catchments(self) -> "CatchmentDataset":
        """Return a new dataset containing only samples with a catchment polygon."""
        filtered = [s for s in self._samples if s.has_catchment]
        return CatchmentDataset(filtered, source_path=self.source_path)

    def filter(
        self,
        *,
        min_result: Optional[float] = None,
        max_result: Optional[float] = None,
        delineation_status: Optional[str] = None,
        site_name_contains: Optional[str] = None,
        after: Union[str, date, None] = None,
        before: Union[str, date, None] = None,
    ) -> "CatchmentDataset":
        """Return a filtered subset of this dataset.

        All supplied criteria are combined with AND logic.

        Parameters
        ----------
        min_result, max_result : float, optional
            Keep only samples whose measurement result falls within
            [*min_result*, *max_result*].
        delineation_status : str, optional
            Keep only samples where ``delineation_status == value``
            (e.g. ``"ok"``).
        site_name_contains : str, optional
            Case-insensitive substring match against ``site_name``.
        after, before : str or date, optional
            Keep only samples with ``date >= after`` and/or ``date <= before``.
        """
        result: List[Sample] = []
        after_date = _to_date_obj(after) if after is not None else None
        before_date = _to_date_obj(before) if before is not None else None

        for s in self._samples:
            if min_result is not None and (s.result is None or s.result < min_result):
                continue
            if max_result is not None and (s.result is None or s.result > max_result):
                continue
            if delineation_status is not None and s.delineation_status != delineation_status:
                continue
            if (
                site_name_contains is not None
                and site_name_contains.lower() not in s.site_name.lower()
            ):
                continue
            if after_date is not None and (s.date is None or s.date < after_date):
                continue
            if before_date is not None and (s.date is None or s.date > before_date):
                continue
            result.append(s)

        return CatchmentDataset(result, source_path=self.source_path)

    # ------------------------------------------------------------------
    # Bulk EO queries
    # ------------------------------------------------------------------

    def query_all_ndvi(
        self,
        connection: "openeo.Connection",
        days_window: int = 15,
        *,
        output_dir: Optional[Union[str, Path]] = None,
        collection: str = SENTINEL2_COLLECTION,
        max_cloud_cover: int = 85,
    ) -> "Dict[str, openeo.DataCube]":
        """Run NDVI queries for every sample that has a catchment polygon.

        Parameters
        ----------
        connection : openeo.Connection
        days_window : int
            Passed to each :meth:`Sample.query_ndvi` call.
        output_dir : Path, optional
            If provided, immediately download each cube as a GeoTIFF into
            this directory (``<notation>_ndvi.tif``).
        collection : str
        max_cloud_cover : int

        Returns
        -------
        dict mapping notation → DataCube
            All cubes (whether downloaded or not).
        """
        output_dir_path = Path(output_dir) if output_dir is not None else None
        if output_dir_path is not None:
            output_dir_path.mkdir(parents=True, exist_ok=True)

        cubes: Dict[str, Any] = {}
        eligible = [s for s in self._samples if s.has_catchment]
        logger.info("Running NDVI queries for %d samples …", len(eligible))

        for s in eligible:
            try:
                cube = s.query_ndvi(
                    connection,
                    days_window=days_window,
                    collection=collection,
                    max_cloud_cover=max_cloud_cover,
                )
                cubes[s.notation] = cube

                if output_dir_path is not None:
                    out_path = output_dir_path / f"{s.notation}_ndvi.tif"
                    logger.info("Downloading NDVI → %s", out_path)
                    cube.download(str(out_path), format="GTiff")
            except Exception as exc:
                logger.warning(
                    "NDVI query failed for '%s' (%s): %s",
                    s.site_name,
                    s.notation,
                    exc,
                )

        return cubes

    def query_all_bands(
        self,
        connection: "openeo.Connection",
        bands: Sequence[str],
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
        *,
        days_window: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        collection: str = SENTINEL2_COLLECTION,
        max_cloud_cover: int = 85,
    ) -> "Dict[str, openeo.DataCube]":
        """Query specified bands for every sample with a catchment polygon.

        Parameters
        ----------
        connection : openeo.Connection
        bands : sequence of str
        start_date, end_date : str or date, optional
        days_window : int, optional
        output_dir : Path, optional
            If provided, each cube is immediately downloaded as
            ``<notation>_<bands>.tif``.
        collection : str
        max_cloud_cover : int

        Returns
        -------
        dict mapping notation → DataCube
        """
        output_dir_path = Path(output_dir) if output_dir is not None else None
        if output_dir_path is not None:
            output_dir_path.mkdir(parents=True, exist_ok=True)

        cubes: Dict[str, Any] = {}
        eligible = [s for s in self._samples if s.has_catchment]
        band_label = "_".join(bands)
        logger.info("Running band query (%s) for %d samples …", band_label, len(eligible))

        for s in eligible:
            try:
                cube = s.query_bands(
                    connection,
                    bands=bands,
                    start_date=start_date,
                    end_date=end_date,
                    days_window=days_window,
                    collection=collection,
                    max_cloud_cover=max_cloud_cover,
                )
                cubes[s.notation] = cube

                if output_dir_path is not None:
                    out_path = output_dir_path / f"{s.notation}_{band_label}.tif"
                    logger.info("Downloading → %s", out_path)
                    cube.download(str(out_path), format="GTiff")
            except Exception as exc:
                logger.warning(
                    "Band query failed for '%s' (%s): %s",
                    s.site_name,
                    s.notation,
                    exc,
                )

        return cubes

    # ------------------------------------------------------------------
    # Export / conversion
    # ------------------------------------------------------------------

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Return the dataset as a GeoDataFrame with catchment polygons.

        The geometry column contains each sample's catchment polygon (or
        ``None`` where delineation failed).
        """
        rows = []
        geometries = []
        for s in self._samples:
            row_dict = s._row.to_dict()
            rows.append(row_dict)
            geometries.append(s.catchment)

        gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")
        return gdf

    def summary(self) -> pd.DataFrame:
        """Return a concise summary DataFrame (one row per sample)."""
        records = []
        for s in self._samples:
            records.append(
                {
                    "notation": s.notation,
                    "site_name": s.site_name,
                    "date": s.date,
                    "result": s.result,
                    "unit": s.unit,
                    "determinand": s.determinand,
                    "latitude": s.latitude,
                    "longitude": s.longitude,
                    "delineation_status": s.delineation_status,
                    "has_catchment": s.has_catchment,
                    "catchment_area_km2": s.catchment_area_km2(),
                    "flow_acc_at_pour_point": s.flow_acc_at_pour_point,
                }
            )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    @overload
    def __getitem__(self, idx: int) -> Sample: ...

    @overload
    def __getitem__(self, idx: slice) -> "CatchmentDataset": ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[Sample, "CatchmentDataset"]:
        if isinstance(idx, slice):
            return CatchmentDataset(self._samples[idx], source_path=self.source_path)
        return self._samples[idx]

    def __repr__(self) -> str:
        n_catch = sum(1 for s in self._samples if s.has_catchment)
        src = f", source={self.source_path.name!r}" if self.source_path else ""
        return f"CatchmentDataset({len(self._samples)} samples, {n_catch} with catchments{src})"

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _to_iso(d: Union[str, date]) -> str:
    """Convert a str or date to an ISO-8601 string."""
    if isinstance(d, str):
        return d[:10]  # accept "YYYY-MM-DD HH:MM:SS" as well
    return d.isoformat()


def _to_date_obj(d: Union[str, date]) -> date:
    """Convert a str or date to a :class:`datetime.date`."""
    if isinstance(d, date):
        return d
    return datetime.strptime(d[:10], "%Y-%m-%d").date()
