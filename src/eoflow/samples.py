"""
samples.py — Catchment-aware water-quality sample store with EO query support.

Provides two main classes:

* :class:`CatchmentSample`
    A single water-quality observation enriched with the delineated catchment
    polygon produced by ``dataset_builder.py``.  Exposes methods to query
    Earth Observation data (Sentinel-2 L2A via openEO) for the catchment area
    over an arbitrary date range, including built-in NDVI / NDWI helpers and
    support for any raw band combination.

* :class:`CatchmentDataset`
    A collection of :class:`CatchmentSample` objects loaded from the
    GeoPackage written by ``dataset_builder.py``.  Supports filtering,
    iteration, and bulk EO queries.

Typical usage
-------------
    >>> from samples import CatchmentDataset
    >>> ds = CatchmentDataset.from_gpkg("data/devon_water_quality_dataset.gpkg")
    >>> print(ds)
    CatchmentDataset(10 samples, 10 with catchments)

    >>> sample = ds[0]
    >>> print(sample)
    CatchmentSample(id='...', site='RIVER OTTER AT DOTTON MILL', date='2020-01-14', result=57.0)

    >>> # Compute NDVI for the catchment in the month around the sample date
    >>> conn = sample.connect_openeo()                  # prompts OIDC login
    >>> ndvi = sample.query_ndvi(conn, days_window=15)  # returns DataCube
    >>> ndvi.download("otter_ndvi.tif", format="GTiff")

    >>> # Query raw bands
    >>> cube = sample.query_bands(conn, ["B03", "B08", "B11"],
    ...                           start_date="2020-01-01",
    ...                           end_date="2020-03-31")

EO backend
----------
All queries target the Copernicus Data Space Ecosystem openEO federation
endpoint (``openeofed.dataspace.copernicus.eu``) using ``SENTINEL2_L2A``
by default.  A free Copernicus Data Space account is required; authentication
is done via OIDC (``connection.authenticate_oidc()``).

References
----------
* openEO Python client: https://open-eo.github.io/openeo-python-client/
* Copernicus Data Space: https://dataspace.copernicus.eu/
"""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, mapping
from shapely.geometry.base import BaseGeometry

from eoflow.log_utils import get_logger

if TYPE_CHECKING:
    import openeo  # type: ignore[import-untyped]

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

# Columns written by dataset_builder that we care about
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
# CatchmentSample
# ---------------------------------------------------------------------------


class CatchmentSample:
    """A water-quality observation with its delineated catchment polygon.

    Attributes
    ----------
    row : pandas.Series
        The raw row from the GeoPackage GeoDataFrame (read-only view).
    catchment : shapely.geometry.Polygon or None
        The delineated catchment polygon in WGS 84 (EPSG:4326), or ``None``
        if delineation failed for this sample.
    """

    def __init__(self, row: pd.Series, catchment: Optional[BaseGeometry]) -> None:
        self._row = row.copy()
        self.catchment: Optional[BaseGeometry] = catchment

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
        try:
            import openeo  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'openeo' package is required for EO queries.  "
                "Install it with:  uv add openeo  (or  pip install openeo)"
            ) from exc

        conn = openeo.connect(backend)
        if authenticate:
            conn.authenticate_oidc()
        return conn

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

        logger.info(
            "Loading %s for site '%s' | bands=%s | %s → %s",
            collection,
            self.site_name,
            bands,
            start_str,
            end_str,
        )

        cube = connection.load_collection(
            collection,
            spatial_extent=spatial_extent,
            temporal_extent=[start_str, end_str],
            bands=list(bands),
            max_cloud_cover=max_cloud_cover,
        )
        return cube

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
        # openEO has no built-in ndwi process; compute with band_math
        green = cube.filter_bands([green_band]).rename_labels(dimension="bands", target=["green"])
        nir = cube.filter_bands([nir_band]).rename_labels(dimension="bands", target=["nir"])
        merged = green.merge_cubes(nir)
        ndwi = merged.reduce_dimension(
            dimension="bands",
            reducer=lambda data: (data.array_element(0) - data.array_element(1))
            / (data.array_element(0) + data.array_element(1)),
        )
        logger.info("NDWI process applied (green=%s, nir=%s)", green_band, nir_band)
        return ndwi

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
        a = cube.filter_bands([band_a]).rename_labels(dimension="bands", target=["a"])
        b = cube.filter_bands([band_b]).rename_labels(dimension="bands", target=["b"])
        merged = a.merge_cubes(b)
        result = merged.reduce_dimension(
            dimension="bands",
            reducer=lambda data: (data.array_element(0) - data.array_element(1))
            / (data.array_element(0) + data.array_element(1)),
        )
        logger.info("%s applied (%s, %s)", label, band_a, band_b)
        return result

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
        nir = cube.filter_bands([nir_b]).rename_labels(dimension="bands", target=["nir"])
        red = cube.filter_bands([red_b]).rename_labels(dimension="bands", target=["red"])
        blue = cube.filter_bands([blue_b]).rename_labels(dimension="bands", target=["blue"])
        merged = nir.merge_cubes(red).merge_cubes(blue)

        def _evi_reducer(data):
            nir_v = data.array_element(0)
            red_v = data.array_element(1)
            blue_v = data.array_element(2)
            # S2 L2A reflectance is scaled ×10000.  Normalised-difference indices
            # (NDVI, NDWI, …) are unaffected because the scale cancels in the
            # ratio, but EVI has an absolute "+1" term in the denominator that
            # must be expressed in the same units (i.e. +10000 for scaled data).
            # Equivalent to the standard formula applied to [0, 1] reflectance.
            return 2.5 * (nir_v - red_v) / (nir_v + 6 * red_v - 7.5 * blue_v + 10000)

        evi = merged.reduce_dimension(dimension="bands", reducer=_evi_reducer)
        logger.info("EVI applied (nir=%s, red=%s, blue=%s)", nir_b, red_b, blue_b)
        return evi

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
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CatchmentSample("
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
            f"CatchmentSample\n"
            f"  Site        : {self.site_name}\n"
            f"  Notation    : {self.notation}\n"
            f"  Date        : {self.date}\n"
            f"  Measurement : {self.result} {self.unit} ({self.determinand})\n"
            f"  Location    : ({self.latitude:.5f}, {self.longitude:.5f})\n"
            f"  Catchment   : {self.delineation_status} — area ≈ {area_str}\n"
            f"  Flow acc    : {self.flow_acc_at_pour_point} cells"
        )


# ---------------------------------------------------------------------------
# CatchmentDataset
# ---------------------------------------------------------------------------


class CatchmentDataset:
    """A collection of :class:`CatchmentSample` objects.

    Load from the GeoPackage produced by ``dataset_builder.py`` with
    :meth:`from_gpkg`, then iterate, filter, and query EO data.

    Parameters
    ----------
    samples : list of CatchmentSample
        The samples in the dataset.
    source_path : Path or None
        Path to the originating GeoPackage (informational).
    """

    def __init__(
        self,
        samples: List[CatchmentSample],
        source_path: Optional[Path] = None,
    ) -> None:
        self._samples: List[CatchmentSample] = samples
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

        The GeoPackage must be in the format written by ``dataset_builder.py``
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

        samples: List[CatchmentSample] = []
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

            samples.append(CatchmentSample(row=row, catchment=catchment))  # type: ignore[arg-type]

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
            GeoDataFrame in the format produced by ``dataset_builder.py``.
        source_path : Path, optional
            Optional path to record as the data source.
        """
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        samples = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            catchment = geom if isinstance(geom, Polygon) else None
            samples.append(CatchmentSample(row=row, catchment=catchment))

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
        result: List[CatchmentSample] = []
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
            Passed to each :meth:`CatchmentSample.query_ndvi` call.
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

    def __iter__(self) -> Iterator[CatchmentSample]:
        return iter(self._samples)

    def __getitem__(self, idx: Union[int, slice]) -> Union[CatchmentSample, "CatchmentDataset"]:
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
