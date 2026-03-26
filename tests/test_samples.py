"""
Unit and integration tests for eoflow.samples.

Structure
---------
* Fixtures            — shared in-memory data builders (no real files required)
* TestSampleProperties   — property accessors, edge cases, NaN handling
* TestSampleSpatial      — bbox, GeoJSON, area, has_catchment
* TestSampleDates        — _resolve_dates logic (days_window, explicit, default)
* TestSampleEOInterface  — query_bands / query_ndvi / query_ndwi / query_index
*                                   all use a mocked openeo Connection & DataCube
* TestSampleConnectOpeneo — connect_openeo import guard and auth flag
* TestCatchmentDatasetFromGpkg    — from_gpkg construction, WKT fallback, only_delineated
* TestCatchmentDatasetFromGeoDataFrame — from_geodataframe construction
* TestCatchmentDatasetFilter      — filter() method, all criteria
* TestCatchmentDatasetCollectionProtocol — len, iter, getitem, slicing
* TestCatchmentDatasetBulkEO      — query_all_ndvi / query_all_bands (mocked openeo)
* TestCatchmentDatasetExport      — to_geodataframe, summary
* TestModulePublicAPI             — importability, __all__, top-level helpers
* TestIntegration                 — slow tests that hit the real GeoPackage on disk
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from eoflow.samples import (
    OPENEO_BACKEND,
    SENTINEL2_COLLECTION,
    CatchmentDataset,
    Sample,
    _to_date_obj,
    _to_iso,
)

# ---------------------------------------------------------------------------
# Paths used by integration tests
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_GPKG_PATH = _DATA_DIR / "devon_water_quality_dataset.gpkg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_polygon(
    lon: float = -3.5,
    lat: float = 50.7,
    size: float = 0.05,
) -> Polygon:
    """Return a small square polygon around (*lon*, *lat*)."""
    h = size / 2
    return Polygon(
        [
            (lon - h, lat - h),
            (lon + h, lat - h),
            (lon + h, lat + h),
            (lon - h, lat + h),
            (lon - h, lat - h),
        ]
    )


def _make_row(
    lat: float = 50.7,
    lon: float = -3.5,
    result: float = 10.0,
    date_str: str = "2020-06-15",
    status: str = "ok",
    notation: str = "SW-00000001",
    site: str = "TEST RIVER AT TEST BRIDGE",
    snap_lat: Optional[float] = 50.701,
    snap_lon: Optional[float] = -3.501,
    flow_acc: Optional[float] = 250.0,
) -> pd.Series:
    """Build a minimal pandas Series matching the GeoPackage schema."""
    data: dict = {
        "id": f"https://example.com/obs/{notation}",
        "samplingPoint.notation": notation,
        "samplingPoint.prefLabel": site,
        "samplingPoint.easting": 292546,
        "samplingPoint.northing": 91506,
        "samplingPoint.region": "SouthWest",
        "samplingPoint.area": "SOUTH WEST - DEVON AND CORNWALL",
        "samplingPoint.subArea": "EAST DEVON AND CORNWALL",
        "phenomenonTime": f"{date_str} 10:00:00",
        "determinand.notation": 6396,
        "determinand.prefLabel": "Turbidity",
        "result": result,
        "unit": "NEPHELOMETRIC TURBIDITY UNITS",
        "Date": date_str,
        "result_numeric": result,
        "longitude": lon,
        "latitude": lat,
        "delineation_status": status,
        "delineation_error": "",
        "snap_longitude": snap_lon if snap_lon is not None else float("nan"),
        "snap_latitude": snap_lat if snap_lat is not None else float("nan"),
        "flow_acc_at_pour_point": flow_acc if flow_acc is not None else float("nan"),
    }
    return pd.Series(data)


def _make_sample(
    polygon: Optional[Polygon] = None,
    **row_kwargs,
) -> Sample:
    """Build a Sample with optional polygon override."""
    if polygon is None:
        polygon = _make_polygon()
    return Sample(row=_make_row(**row_kwargs), catchment=polygon)


def _make_mock_connection() -> MagicMock:
    """Return a mock openEO Connection whose load_collection returns a mock DataCube."""
    conn = MagicMock()
    cube = MagicMock()
    # load_collection → cube; ndvi → cube; filter_bands → cube; etc.
    conn.load_collection.return_value = cube
    cube.ndvi.return_value = cube
    cube.filter_bands.return_value = cube
    cube.rename_labels.return_value = cube
    cube.merge_cubes.return_value = cube
    cube.reduce_dimension.return_value = cube
    return conn


def _make_gpkg(path: Path, n: int = 3) -> Path:
    """Write a minimal GeoPackage to *path* and return it."""
    polygons = [_make_polygon(lon=-3.5 - i * 0.1, lat=50.7 + i * 0.05) for i in range(n)]
    rows = [
        _make_row(
            lat=50.7 + i * 0.05,
            lon=-3.5 - i * 0.1,
            notation=f"SW-{i:08d}",
            result=float(i + 1) * 5,
            date_str=f"2020-0{i + 1}-15",
        )
        for i in range(n)
    ]
    gdf = gpd.GeoDataFrame(
        [r.to_dict() for r in rows],
        geometry=polygons,
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(path), driver="GPKG")
    return path


# ===========================================================================
# Sample – property accessors
# ===========================================================================


class TestSampleProperties:
    """Property accessors return correct types and handle missing / NaN values."""

    def test_id(self):
        s = _make_sample(notation="SW-12345678")
        assert "SW-12345678" in s.id

    def test_site_name(self):
        s = _make_sample(site="RIVER EXE AT EXETER")
        assert s.site_name == "RIVER EXE AT EXETER"

    def test_notation(self):
        s = _make_sample(notation="SW-99999999")
        assert s.notation == "SW-99999999"

    def test_latitude(self):
        s = _make_sample(lat=50.72)
        assert s.latitude == pytest.approx(50.72)

    def test_longitude(self):
        s = _make_sample(lon=-3.53)
        assert s.longitude == pytest.approx(-3.53)

    def test_result(self):
        s = _make_sample(result=42.5)
        assert s.result == pytest.approx(42.5)

    def test_unit(self):
        s = _make_sample()
        assert "TURBIDITY" in s.unit.upper()

    def test_determinand(self):
        s = _make_sample()
        assert s.determinand == "Turbidity"

    def test_date_parsed(self):
        s = _make_sample(date_str="2021-03-25")
        assert s.date == date(2021, 3, 25)

    def test_date_only_uses_first_10_chars(self):
        row = _make_row(date_str="2020-06-15")
        row["Date"] = "2020-06-15 10:30:00"
        s = Sample(row=row, catchment=_make_polygon())
        assert s.date == date(2020, 6, 15)

    def test_date_none_when_missing(self):
        row = _make_row()
        row["Date"] = None
        s = Sample(row=row, catchment=_make_polygon())
        assert s.date is None

    def test_date_none_when_nan_float(self):
        row = _make_row()
        row["Date"] = float("nan")
        s = Sample(row=row, catchment=_make_polygon())
        assert s.date is None

    def test_delineation_status(self):
        s = _make_sample(status="ok")
        assert s.delineation_status == "ok"

    def test_delineation_error_empty_by_default(self):
        s = _make_sample()
        assert s.delineation_error == ""

    def test_snap_latitude(self):
        s = _make_sample(snap_lat=50.705)
        assert s.snap_latitude == pytest.approx(50.705)

    def test_snap_longitude(self):
        s = _make_sample(snap_lon=-3.502)
        assert s.snap_longitude == pytest.approx(-3.502)

    def test_snap_latitude_none_when_missing(self):
        s = _make_sample(snap_lat=None)
        assert s.snap_latitude is None

    def test_snap_longitude_none_when_missing(self):
        s = _make_sample(snap_lon=None)
        assert s.snap_longitude is None

    def test_flow_acc_at_pour_point(self):
        s = _make_sample(flow_acc=315.0)
        assert s.flow_acc_at_pour_point == pytest.approx(315.0)

    def test_flow_acc_none_when_missing(self):
        s = _make_sample(flow_acc=None)
        assert s.flow_acc_at_pour_point is None

    def test_latitude_none_when_nan(self):
        row = _make_row()
        row["latitude"] = float("nan")
        s = Sample(row=row, catchment=_make_polygon())
        assert s.latitude is None

    def test_longitude_none_when_nan(self):
        row = _make_row()
        row["longitude"] = float("nan")
        s = Sample(row=row, catchment=_make_polygon())
        assert s.longitude is None

    def test_result_none_when_nan(self):
        row = _make_row()
        row["result"] = float("nan")
        s = Sample(row=row, catchment=_make_polygon())
        assert s.result is None

    def test_sample_point_returns_shapely_point(self):
        s = _make_sample(lat=50.7, lon=-3.5)
        pt = s.sample_point
        assert isinstance(pt, Point)
        assert pt.x == pytest.approx(-3.5)
        assert pt.y == pytest.approx(50.7)

    def test_sample_point_none_when_no_coords(self):
        row = _make_row()
        row["latitude"] = float("nan")
        row["longitude"] = float("nan")
        s = Sample(row=row, catchment=_make_polygon())
        assert s.sample_point is None

    def test_snapped_pour_point_returns_point(self):
        s = _make_sample(snap_lat=50.701, snap_lon=-3.501)
        pt = s.snapped_pour_point
        assert isinstance(pt, Point)
        assert pt.x == pytest.approx(-3.501)
        assert pt.y == pytest.approx(50.701)

    def test_snapped_pour_point_none_when_missing(self):
        s = _make_sample(snap_lat=None, snap_lon=None)
        assert s.snapped_pour_point is None

    def test_repr_contains_site_and_date(self):
        s = _make_sample(site="RIVER EXE", date_str="2021-07-01")
        r = repr(s)
        assert "RIVER EXE" in r
        # repr shows datetime.date(2021, 7, 1) rather than the ISO string
        assert "2021" in r and "7" in r and "1" in r

    def test_str_contains_key_fields(self):
        s = _make_sample(site="RIVER EXE", result=42.0)
        text = str(s)
        assert "RIVER EXE" in text
        assert "42.0" in text


# ===========================================================================
# Sample – spatial helpers
# ===========================================================================


class TestSampleSpatial:
    def test_has_catchment_true_when_polygon_set(self):
        s = _make_sample(polygon=_make_polygon())
        assert s.has_catchment is True

    def test_has_catchment_false_when_none(self):
        s = Sample(row=_make_row(), catchment=None)
        assert s.has_catchment is False

    def test_has_catchment_false_when_empty_polygon(self):
        s = Sample(row=_make_row(), catchment=Polygon())
        assert s.has_catchment is False

    def test_catchment_bbox_returns_four_floats(self):
        poly = _make_polygon(lon=-3.5, lat=50.7, size=0.1)
        s = Sample(row=_make_row(), catchment=poly)
        bbox = s.catchment_bbox()
        assert bbox is not None
        west, south, east, north = bbox
        assert west < east
        assert south < north

    def test_catchment_bbox_approximate_extent(self):
        poly = _make_polygon(lon=-3.5, lat=50.7, size=0.1)
        s = Sample(row=_make_row(), catchment=poly)
        bbox = s.catchment_bbox()
        assert bbox is not None
        west, south, east, north = bbox
        assert west == pytest.approx(-3.55, abs=1e-6)
        assert east == pytest.approx(-3.45, abs=1e-6)
        assert south == pytest.approx(50.65, abs=1e-6)
        assert north == pytest.approx(50.75, abs=1e-6)

    def test_catchment_bbox_none_when_no_catchment(self):
        s = Sample(row=_make_row(), catchment=None)
        assert s.catchment_bbox() is None

    def test_catchment_geojson_returns_dict(self):
        s = _make_sample()
        geojson = s.catchment_geojson()
        assert isinstance(geojson, dict)
        assert geojson["type"] == "Polygon"

    def test_catchment_geojson_none_when_no_catchment(self):
        s = Sample(row=_make_row(), catchment=None)
        assert s.catchment_geojson() is None

    def test_catchment_area_km2_positive(self):
        s = _make_sample()
        area = s.catchment_area_km2()
        assert area is not None
        assert area > 0

    def test_catchment_area_km2_reasonable_for_small_polygon(self):
        # 0.1° × 0.1° box around lat=50.7 → roughly 50 km²
        poly = _make_polygon(lon=-3.5, lat=50.7, size=0.1)
        s = Sample(row=_make_row(lat=50.7, lon=-3.5), catchment=poly)
        area = s.catchment_area_km2()
        assert area is not None
        assert 30 < area < 80  # sanity bounds

    def test_catchment_area_km2_none_when_no_catchment(self):
        s = Sample(row=_make_row(), catchment=None)
        assert s.catchment_area_km2() is None

    def test_catchment_accepts_multipolygon(self):
        """MultiPolygon is a valid BaseGeometry and should be stored."""
        p1 = _make_polygon(lon=-3.5, lat=50.7, size=0.02)
        p2 = _make_polygon(lon=-3.6, lat=50.8, size=0.02)
        mp = MultiPolygon([p1, p2])
        s = Sample(row=_make_row(), catchment=mp)
        assert s.has_catchment is True
        assert s.catchment_bbox() is not None


# ===========================================================================
# Sample – date resolution
# ===========================================================================


class TestSampleDates:
    def test_resolve_dates_explicit(self):
        s = _make_sample(date_str="2020-06-15")
        start, end = s._resolve_dates("2020-01-01", "2020-12-31", None)
        assert start == "2020-01-01"
        assert end == "2020-12-31"

    def test_resolve_dates_days_window(self):
        s = _make_sample(date_str="2020-06-15")
        start, end = s._resolve_dates(None, None, 10)
        expected_start = date(2020, 6, 5)
        expected_end = date(2020, 6, 25)
        assert start == expected_start.isoformat()
        assert end == expected_end.isoformat()

    def test_resolve_dates_days_window_zero(self):
        s = _make_sample(date_str="2020-06-15")
        start, end = s._resolve_dates(None, None, 0)
        assert start == end == "2020-06-15"

    def test_resolve_dates_days_window_overrides_explicit(self):
        s = _make_sample(date_str="2020-06-15")
        # days_window takes priority
        start, end = s._resolve_dates("2019-01-01", "2019-12-31", 5)
        assert start == date(2020, 6, 10).isoformat()
        assert end == date(2020, 6, 20).isoformat()

    def test_resolve_dates_default_window_30_days(self):
        s = _make_sample(date_str="2020-06-15")
        start, end = s._resolve_dates(None, None, None)
        assert start == date(2020, 5, 16).isoformat()
        assert end == date(2020, 7, 15).isoformat()

    def test_resolve_dates_raises_without_date_and_no_window(self):
        row = _make_row()
        row["Date"] = None
        s = Sample(row=row, catchment=_make_polygon())
        with pytest.raises(ValueError, match="no date"):
            s._resolve_dates(None, None, None)

    def test_resolve_dates_raises_days_window_without_sample_date(self):
        row = _make_row()
        row["Date"] = None
        s = Sample(row=row, catchment=_make_polygon())
        with pytest.raises(ValueError, match="days_window"):
            s._resolve_dates(None, None, 15)

    def test_resolve_dates_accepts_date_objects(self):
        s = _make_sample(date_str="2020-06-15")
        start, end = s._resolve_dates(date(2020, 1, 1), date(2020, 6, 30), None)
        assert start == "2020-01-01"
        assert end == "2020-06-30"

    def test_resolve_dates_truncates_datetime_strings(self):
        s = _make_sample(date_str="2020-06-15")
        start, end = s._resolve_dates("2020-01-01 08:00:00", "2020-12-31 23:59:59", None)
        assert start == "2020-01-01"
        assert end == "2020-12-31"


# ===========================================================================
# Sample – EO query interface (mocked openeo)
# ===========================================================================


class TestSampleEOInterface:
    """All openEO calls use a mocked Connection — no network required."""

    @pytest.fixture()
    def sample(self):
        return _make_sample(date_str="2020-06-15", lat=50.7, lon=-3.5)

    @pytest.fixture()
    def conn(self):
        return _make_mock_connection()

    def test_query_bands_calls_load_collection(self, sample, conn):
        sample.query_bands(conn, ["B04", "B08"], "2020-06-01", "2020-06-30")
        conn.load_collection.assert_called_once()
        call_kwargs = conn.load_collection.call_args
        assert call_kwargs[0][0] == SENTINEL2_COLLECTION
        assert "B04" in call_kwargs[1]["bands"]
        assert "B08" in call_kwargs[1]["bands"]

    def test_query_bands_temporal_extent(self, sample, conn):
        sample.query_bands(conn, ["B04"], "2020-06-01", "2020-07-31")
        kwargs = conn.load_collection.call_args[1]
        assert kwargs["temporal_extent"] == ["2020-06-01", "2020-07-31"]

    def test_query_bands_spatial_extent_from_catchment(self, sample, conn):
        sample.query_bands(conn, ["B04"], "2020-06-01", "2020-06-30")
        kwargs = conn.load_collection.call_args[1]
        bbox = kwargs["spatial_extent"]
        assert "west" in bbox and "south" in bbox
        assert "east" in bbox and "north" in bbox
        assert bbox["west"] < bbox["east"]
        assert bbox["south"] < bbox["north"]

    def test_query_bands_max_cloud_cover_forwarded(self, sample, conn):
        sample.query_bands(conn, ["B04"], "2020-06-01", "2020-06-30", max_cloud_cover=50)
        kwargs = conn.load_collection.call_args[1]
        assert kwargs["max_cloud_cover"] == 50

    def test_query_bands_custom_collection(self, sample, conn):
        sample.query_bands(conn, ["B04"], "2020-06-01", "2020-06-30", collection="SENTINEL2_L1C")
        assert conn.load_collection.call_args[0][0] == "SENTINEL2_L1C"

    def test_query_bands_raises_without_catchment(self, conn):
        s = Sample(row=_make_row(), catchment=None)
        with pytest.raises(ValueError, match="catchment polygon"):
            s.query_bands(conn, ["B04"], "2020-01-01", "2020-12-31")

    def test_query_bands_uses_days_window(self, sample, conn):
        sample.query_bands(conn, ["B04"], days_window=7)
        kwargs = conn.load_collection.call_args[1]
        start_str, end_str = kwargs["temporal_extent"]
        start = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)
        assert (end - start).days == 14  # 2 × 7

    def test_query_ndvi_calls_ndvi_process(self, sample, conn):
        sample.query_ndvi(conn, "2020-06-01", "2020-06-30")
        # underlying cube's ndvi() should have been called
        cube = conn.load_collection.return_value
        cube.ndvi.assert_called_once()
        call_kwargs = cube.ndvi.call_args[1]
        assert call_kwargs["nir"] == "B08"
        assert call_kwargs["red"] == "B04"

    def test_query_ndvi_custom_bands(self, sample, conn):
        sample.query_ndvi(conn, "2020-06-01", "2020-06-30", nir_band="B8A", red_band="B04")
        cube = conn.load_collection.return_value
        call_kwargs = cube.ndvi.call_args[1]
        assert call_kwargs["nir"] == "B8A"
        assert call_kwargs["red"] == "B04"

    def test_query_ndwi_loads_green_and_nir(self, sample, conn):
        sample.query_ndwi(conn, "2020-06-01", "2020-06-30")
        kwargs = conn.load_collection.call_args[1]
        assert "B03" in kwargs["bands"]
        assert "B08" in kwargs["bands"]

    def test_query_index_ndvi(self, sample, conn):
        result = sample.query_index(conn, "NDVI", "2020-06-01", "2020-06-30")
        assert result is not None
        conn.load_collection.assert_called()

    def test_query_index_ndwi(self, sample, conn):
        result = sample.query_index(conn, "NDWI", "2020-06-01", "2020-06-30")
        assert result is not None

    def test_query_index_mndwi(self, sample, conn):
        result = sample.query_index(conn, "MNDWI", "2020-06-01", "2020-06-30")
        assert result is not None

    def test_query_index_ndre(self, sample, conn):
        result = sample.query_index(conn, "NDRE", "2020-06-01", "2020-06-30")
        assert result is not None

    def test_query_index_evi(self, sample, conn):
        result = sample.query_index(conn, "EVI", "2020-06-01", "2020-06-30")
        assert result is not None

    def test_query_index_case_insensitive(self, sample, conn):
        # lowercase should work
        result = sample.query_index(conn, "ndvi", "2020-06-01", "2020-06-30")
        assert result is not None

    def test_query_index_unknown_raises_value_error(self, sample, conn):
        with pytest.raises(ValueError, match="FOOBAR"):
            sample.query_index(conn, "FOOBAR", "2020-06-01", "2020-06-30")

    def test_query_index_error_message_lists_supported(self, sample, conn):
        with pytest.raises(ValueError, match="NDVI"):
            sample.query_index(conn, "UNKNOWN", "2020-06-01", "2020-06-30")

    def test_spatial_extent_dict_matches_bbox(self, sample):
        extent = sample._spatial_extent_dict()
        bbox = sample.catchment_bbox()
        assert extent["west"] == pytest.approx(bbox[0])
        assert extent["south"] == pytest.approx(bbox[1])
        assert extent["east"] == pytest.approx(bbox[2])
        assert extent["north"] == pytest.approx(bbox[3])

    def test_spatial_extent_dict_raises_without_catchment(self):
        s = Sample(row=_make_row(), catchment=None)
        with pytest.raises(ValueError):
            s._spatial_extent_dict()


# ===========================================================================
# Sample – connect_openeo
# ===========================================================================


class TestSampleConnectOpeneo:
    def test_connect_openeo_raises_import_error_when_openeo_missing(self):
        import sys

        # Temporarily hide openeo from imports
        openeo_mod = sys.modules.pop("openeo", None)
        try:
            with pytest.raises(ImportError, match="openeo"):
                # Force the lazy import inside connect_openeo to fail
                with patch.dict("sys.modules", {"openeo": None}):
                    Sample.connect_openeo(authenticate=False)
        finally:
            if openeo_mod is not None:
                sys.modules["openeo"] = openeo_mod

    def test_connect_openeo_does_not_authenticate_when_flag_false(self):
        mock_openeo = MagicMock()
        mock_conn = MagicMock()
        mock_openeo.connect.return_value = mock_conn
        with patch.dict("sys.modules", {"openeo": mock_openeo}):
            # Re-import to pick up patched module
            import importlib

            import eoflow.samples as sm

            importlib.reload(sm)
            _ = sm.Sample.connect_openeo(authenticate=False)
            mock_conn.authenticate_oidc.assert_not_called()

    def test_connect_openeo_default_backend(self):
        mock_openeo = MagicMock()
        mock_openeo.connect.return_value = MagicMock()
        with patch.dict("sys.modules", {"openeo": mock_openeo}):
            import importlib

            import eoflow.samples as sm

            importlib.reload(sm)
            sm.Sample.connect_openeo(authenticate=False)
            mock_openeo.connect.assert_called_with(OPENEO_BACKEND)


# ===========================================================================
# CatchmentDataset – from_gpkg
# ===========================================================================


class TestCatchmentDatasetFromGpkg:
    def test_from_gpkg_basic(self, tmp_path):
        gpkg = _make_gpkg(tmp_path / "test.gpkg", n=3)
        ds = CatchmentDataset.from_gpkg(gpkg)
        assert len(ds) == 3

    def test_from_gpkg_sets_source_path(self, tmp_path):
        gpkg = _make_gpkg(tmp_path / "test.gpkg", n=2)
        ds = CatchmentDataset.from_gpkg(gpkg)
        assert ds.source_path == gpkg

    def test_from_gpkg_all_have_catchments(self, tmp_path):
        gpkg = _make_gpkg(tmp_path / "test.gpkg", n=4)
        ds = CatchmentDataset.from_gpkg(gpkg)
        assert all(s.has_catchment for s in ds)

    def test_from_gpkg_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            CatchmentDataset.from_gpkg(Path("/nonexistent/path.gpkg"))

    def test_from_gpkg_only_delineated_filters(self, tmp_path):
        # Build a GeoPackage with one failed row
        polygons = [_make_polygon(), None]
        rows = [
            _make_row(notation="SW-00000001", status="ok"),
            _make_row(notation="SW-00000002", status="error"),
        ]
        gdf = gpd.GeoDataFrame(
            [r.to_dict() for r in rows],
            geometry=polygons,
            crs="EPSG:4326",
        )
        gpkg = tmp_path / "mixed.gpkg"
        gdf.to_file(str(gpkg), driver="GPKG")

        ds_all = CatchmentDataset.from_gpkg(gpkg, only_delineated=False)
        ds_ok = CatchmentDataset.from_gpkg(gpkg, only_delineated=True)
        assert len(ds_all) == 2
        assert len(ds_ok) == 1
        assert ds_ok[0].delineation_status == "ok"

    def test_from_gpkg_wkt_fallback(self, tmp_path):
        """If geometry column is None but __catchment_wkt is present, use WKT."""
        poly = _make_polygon()
        row = _make_row(notation="SW-WKT00001", status="ok")
        row_dict = row.to_dict()
        row_dict["__catchment_wkt"] = poly.wkt
        # Write with None geometry (simulating failed geometry column)
        gdf = gpd.GeoDataFrame([row_dict], geometry=[None], crs="EPSG:4326")  # type: ignore[arg-type]
        gpkg = tmp_path / "wkt_fallback.gpkg"
        gdf.to_file(str(gpkg), driver="GPKG")

        ds = CatchmentDataset.from_gpkg(gpkg)
        # Should have recovered the polygon from WKT
        assert ds[0].has_catchment

    def test_from_gpkg_samples_are_catchment_sample_instances(self, tmp_path):
        gpkg = _make_gpkg(tmp_path / "test.gpkg", n=2)
        ds = CatchmentDataset.from_gpkg(gpkg)
        for s in ds:
            assert type(s).__name__ == "Sample"

    def test_from_gpkg_reproj_to_wgs84(self, tmp_path):
        """GeoPackages in other CRS should be reprojected transparently."""
        poly = _make_polygon()
        gdf = gpd.GeoDataFrame(
            [_make_row().to_dict()],
            geometry=[poly],
            crs="EPSG:4326",
        ).to_crs("EPSG:27700")  # BNG
        gpkg = tmp_path / "bng.gpkg"
        gdf.to_file(str(gpkg), driver="GPKG")

        ds = CatchmentDataset.from_gpkg(gpkg)
        assert len(ds) == 1
        # After reprojection the catchment should still be valid
        assert ds[0].has_catchment


# ===========================================================================
# CatchmentDataset – from_geodataframe
# ===========================================================================


class TestCatchmentDatasetFromGeoDataFrame:
    def _make_gdf(self, n: int = 3) -> gpd.GeoDataFrame:
        polygons = [_make_polygon(lon=-3.5 - i * 0.1, lat=50.7 + i * 0.05) for i in range(n)]
        rows = [_make_row(notation=f"SW-{i:08d}").to_dict() for i in range(n)]
        return gpd.GeoDataFrame(rows, geometry=polygons, crs="EPSG:4326")

    def test_basic_construction(self):
        gdf = self._make_gdf(3)
        ds = CatchmentDataset.from_geodataframe(gdf)
        assert len(ds) == 3

    def test_all_samples_have_catchments(self):
        gdf = self._make_gdf(2)
        ds = CatchmentDataset.from_geodataframe(gdf)
        assert all(s.has_catchment for s in ds)

    def test_source_path_stored(self, tmp_path):
        gdf = self._make_gdf(1)
        p = tmp_path / "source.gpkg"
        ds = CatchmentDataset.from_geodataframe(gdf, source_path=p)
        assert ds.source_path == p

    def test_reprojection_applied(self):
        gdf = self._make_gdf(1).to_crs("EPSG:27700")
        ds = CatchmentDataset.from_geodataframe(gdf)
        assert len(ds) == 1


# ===========================================================================
# CatchmentDataset – filter
# ===========================================================================


class TestCatchmentDatasetFilter:
    @pytest.fixture()
    def ds(self, tmp_path) -> CatchmentDataset:
        gpkg = _make_gpkg(tmp_path / "filter.gpkg", n=5)
        return CatchmentDataset.from_gpkg(gpkg)

    def test_with_catchments_keeps_only_delineated(self):
        samples = [
            Sample(row=_make_row(notation="SW-00000001", status="ok"), catchment=_make_polygon()),
            Sample(row=_make_row(notation="SW-00000002", status="error"), catchment=None),
        ]
        ds = CatchmentDataset(samples)
        result = ds.with_catchments()
        assert len(result) == 1
        assert result[0].has_catchment

    def test_filter_min_result(self, ds):
        all_results = [s.result for s in ds if s.result is not None]
        mid = sorted(all_results)[len(all_results) // 2]
        filtered = ds.filter(min_result=mid)
        assert all(s.result >= mid for s in filtered if s.result is not None)

    def test_filter_max_result(self, ds):
        all_results = [s.result for s in ds if s.result is not None]
        mid = sorted(all_results)[len(all_results) // 2]
        filtered = ds.filter(max_result=mid)
        assert all(s.result <= mid for s in filtered if s.result is not None)

    def test_filter_min_and_max_result(self, ds):
        filtered = ds.filter(min_result=5.0, max_result=15.0)
        for s in filtered:
            if s.result is not None:
                assert 5.0 <= s.result <= 15.0

    def test_filter_delineation_status(self):
        samples = [
            Sample(row=_make_row(notation="A", status="ok"), catchment=_make_polygon()),
            Sample(row=_make_row(notation="B", status="error"), catchment=None),
        ]
        ds = CatchmentDataset(samples)
        ok_only = ds.filter(delineation_status="ok")
        assert len(ok_only) == 1
        assert ok_only[0].delineation_status == "ok"

    def test_filter_site_name_contains_case_insensitive(self):
        samples = [
            Sample(row=_make_row(site="RIVER EXE AT EXETER"), catchment=_make_polygon()),
            Sample(row=_make_row(site="RIVER DART AT TOTNES"), catchment=_make_polygon()),
        ]
        ds = CatchmentDataset(samples)
        result = ds.filter(site_name_contains="exe")
        assert len(result) == 1
        assert "EXE" in result[0].site_name

    def test_filter_after_date(self):
        samples = [
            Sample(row=_make_row(date_str="2020-01-01"), catchment=_make_polygon()),
            Sample(row=_make_row(date_str="2021-06-15"), catchment=_make_polygon()),
        ]
        ds = CatchmentDataset(samples)
        result = ds.filter(after="2021-01-01")
        assert len(result) == 1
        assert result[0].date == date(2021, 6, 15)

    def test_filter_before_date(self):
        samples = [
            Sample(row=_make_row(date_str="2020-01-01"), catchment=_make_polygon()),
            Sample(row=_make_row(date_str="2021-06-15"), catchment=_make_polygon()),
        ]
        ds = CatchmentDataset(samples)
        result = ds.filter(before="2020-12-31")
        assert len(result) == 1
        assert result[0].date == date(2020, 1, 1)

    def test_filter_combined_criteria(self, ds):
        """AND logic: min_result AND delineation_status=ok."""
        result = ds.filter(min_result=0.0, delineation_status="ok")
        # All generated rows have status="ok"
        assert len(result) == len(ds)

    def test_filter_empty_result(self, ds):
        result = ds.filter(min_result=1e9)  # impossibly high
        assert len(result) == 0

    def test_filter_returns_new_dataset_not_same(self, ds):
        result = ds.filter(min_result=0.0)
        assert result is not ds

    def test_filter_source_path_preserved(self, tmp_path):
        gpkg = _make_gpkg(tmp_path / "filter.gpkg", n=3)
        ds = CatchmentDataset.from_gpkg(gpkg)
        result = ds.filter(min_result=0.0)
        assert result.source_path == ds.source_path


# ===========================================================================
# CatchmentDataset – collection protocol
# ===========================================================================


class TestCatchmentDatasetCollectionProtocol:
    @pytest.fixture()
    def ds(self) -> CatchmentDataset:
        samples = [
            Sample(row=_make_row(notation=f"SW-{i:08d}"), catchment=_make_polygon())
            for i in range(5)
        ]
        return CatchmentDataset(samples)

    def test_len(self, ds):
        assert len(ds) == 5

    def test_iter_yields_catchment_samples(self, ds):
        for s in ds:
            assert isinstance(s, Sample)

    def test_iter_count(self, ds):
        assert sum(1 for _ in ds) == 5

    def test_getitem_integer(self, ds):
        s = ds[0]
        assert isinstance(s, Sample)

    def test_getitem_negative_index(self, ds):
        s = ds[-1]
        assert isinstance(s, Sample)

    def test_getitem_slice_returns_dataset(self, ds):
        sub = ds[1:3]
        assert type(sub).__name__ == "CatchmentDataset"
        assert len(sub) == 2

    def test_getitem_slice_preserves_source_path(self, tmp_path):
        gpkg = _make_gpkg(tmp_path / "proto.gpkg", n=4)
        ds = CatchmentDataset.from_gpkg(gpkg)
        sub = ds[0:2]
        assert sub.source_path == ds.source_path

    def test_repr_contains_counts(self, ds):
        r = repr(ds)
        assert "5" in r
        assert "CatchmentDataset" in r

    def test_str_same_as_repr(self, ds):
        assert str(ds) == repr(ds)

    def test_empty_dataset(self):
        ds = CatchmentDataset([])
        assert len(ds) == 0
        assert list(ds) == []


# ===========================================================================
# CatchmentDataset – bulk EO queries (mocked openeo)
# ===========================================================================


class TestCatchmentDatasetBulkEO:
    @pytest.fixture()
    def ds(self) -> CatchmentDataset:
        samples = [
            Sample(
                row=_make_row(notation=f"SW-{i:08d}", date_str="2020-06-15"),
                catchment=_make_polygon(),
            )
            for i in range(3)
        ]
        return CatchmentDataset(samples)

    @pytest.fixture()
    def conn(self):
        return _make_mock_connection()

    def test_query_all_ndvi_returns_dict(self, ds, conn):
        cubes = ds.query_all_ndvi(conn, days_window=15)
        assert isinstance(cubes, dict)
        assert len(cubes) == len(ds)

    def test_query_all_ndvi_keys_are_notations(self, ds, conn):
        cubes = ds.query_all_ndvi(conn, days_window=15)
        for s in ds:
            assert s.notation in cubes

    def test_query_all_ndvi_skips_no_catchment(self, conn):
        samples = [
            Sample(
                row=_make_row(notation="SW-OK00001", date_str="2020-06-15"),
                catchment=_make_polygon(),
            ),
            Sample(row=_make_row(notation="SW-FAIL001", date_str="2020-06-15"), catchment=None),
        ]
        ds = CatchmentDataset(samples)
        cubes = ds.query_all_ndvi(conn, days_window=15)
        assert "SW-OK00001" in cubes
        assert "SW-FAIL001" not in cubes

    def test_query_all_ndvi_failure_does_not_raise(self, ds):
        conn = MagicMock()
        conn.load_collection.side_effect = RuntimeError("backend down")
        cubes = ds.query_all_ndvi(conn, days_window=15)
        assert isinstance(cubes, dict)
        assert len(cubes) == 0  # all failed gracefully

    def test_query_all_bands_returns_dict(self, ds, conn):
        cubes = ds.query_all_bands(conn, ["B04", "B08"], "2020-06-01", "2020-06-30")
        assert isinstance(cubes, dict)
        assert len(cubes) == len(ds)

    def test_query_all_bands_keys_are_notations(self, ds, conn):
        cubes = ds.query_all_bands(conn, ["B04"], "2020-06-01", "2020-06-30")
        for s in ds:
            assert s.notation in cubes

    def test_query_all_ndvi_downloads_when_output_dir_set(self, ds, conn, tmp_path):
        ds.query_all_ndvi(conn, days_window=15, output_dir=tmp_path)
        cube = conn.load_collection.return_value
        # download() should have been called once per eligible sample
        assert cube.ndvi.return_value.download.call_count == len(ds)

    def test_query_all_bands_downloads_when_output_dir_set(self, ds, conn, tmp_path):
        ds.query_all_bands(conn, ["B04", "B08"], "2020-06-01", "2020-06-30", output_dir=tmp_path)
        cube = conn.load_collection.return_value
        assert cube.download.call_count == len(ds)


# ===========================================================================
# CatchmentDataset – export
# ===========================================================================


class TestCatchmentDatasetExport:
    @pytest.fixture()
    def ds(self) -> CatchmentDataset:
        samples = [
            Sample(
                row=_make_row(notation=f"SW-{i:08d}", result=float(i * 3 + 1)),
                catchment=_make_polygon(),
            )
            for i in range(4)
        ]
        return CatchmentDataset(samples)

    def test_to_geodataframe_returns_gdf(self, ds):
        gdf = ds.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_to_geodataframe_row_count(self, ds):
        gdf = ds.to_geodataframe()
        assert len(gdf) == len(ds)

    def test_to_geodataframe_has_geometry(self, ds):
        gdf = ds.to_geodataframe()
        assert gdf.geometry.notna().all()

    def test_to_geodataframe_crs_is_wgs84(self, ds):
        gdf = ds.to_geodataframe()
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_summary_returns_dataframe(self, ds):
        df = ds.summary()
        assert isinstance(df, pd.DataFrame)

    def test_summary_row_count(self, ds):
        df = ds.summary()
        assert len(df) == len(ds)

    def test_summary_expected_columns(self, ds):
        df = ds.summary()
        for col in (
            "notation",
            "site_name",
            "date",
            "result",
            "has_catchment",
            "catchment_area_km2",
            "flow_acc_at_pour_point",
        ):
            assert col in df.columns, f"Missing column: {col}"

    def test_summary_has_catchment_column_is_bool(self, ds):
        df = ds.summary()
        assert df["has_catchment"].dtype == bool

    def test_summary_catchment_area_positive(self, ds):
        df = ds.summary()
        assert (df["catchment_area_km2"].dropna() > 0).all()

    def test_summary_none_catchment_has_nan_area(self):
        samples = [
            Sample(row=_make_row(notation="SW-NO000001"), catchment=None),
        ]
        ds = CatchmentDataset(samples)
        df = ds.summary()
        assert pd.isna(df["catchment_area_km2"].iloc[0])


# ===========================================================================
# Module-level public API
# ===========================================================================


class TestModulePublicAPI:
    def test_module_importable(self):
        import eoflow.samples  # noqa: F401

    def test_top_level_eoflow_exports(self):
        from eoflow import CatchmentDataset, Sample  # noqa: F401

    def test_constants_defined(self):
        assert isinstance(OPENEO_BACKEND, str)
        assert isinstance(SENTINEL2_COLLECTION, str)
        assert "sentinel2" in SENTINEL2_COLLECTION.lower()

    def test_to_iso_from_string(self):
        assert _to_iso("2020-06-15") == "2020-06-15"

    def test_to_iso_truncates_datetime_string(self):
        assert _to_iso("2020-06-15 12:00:00") == "2020-06-15"

    def test_to_iso_from_date_object(self):
        assert _to_iso(date(2020, 6, 15)) == "2020-06-15"

    def test_to_date_obj_from_string(self):
        assert _to_date_obj("2020-06-15") == date(2020, 6, 15)

    def test_to_date_obj_from_date(self):
        d = date(2020, 6, 15)
        assert _to_date_obj(d) is d

    def test_to_date_obj_truncates_datetime_string(self):
        assert _to_date_obj("2020-06-15 08:30:00") == date(2020, 6, 15)

    def test_catchment_sample_docstring_present(self):
        assert Sample.__doc__ is not None

    def test_catchment_dataset_docstring_present(self):
        assert CatchmentDataset.__doc__ is not None


# ===========================================================================
# Integration tests — require real GeoPackage on disk
# ===========================================================================


def _skip_if_gpkg_missing():
    if not _GPKG_PATH.exists():
        pytest.skip(f"Real GeoPackage not found at {_GPKG_PATH}")


@pytest.mark.integration
class TestIntegration:
    """Tests that load the actual devon_water_quality_dataset.gpkg.

    Run with:   pytest -m integration
    Skip with:  pytest -m "not integration"
    """

    @pytest.fixture(autouse=True)
    def require_gpkg(self):
        _skip_if_gpkg_missing()

    @pytest.fixture()
    def ds(self) -> CatchmentDataset:
        return CatchmentDataset.from_gpkg(_GPKG_PATH)

    def test_loads_without_error(self, ds):
        assert len(ds) > 0

    def test_all_samples_are_catchment_sample_instances(self, ds):
        for s in ds:
            assert type(s).__name__ == "Sample"

    def test_all_delineated_have_catchment(self, ds):
        for s in ds:
            if s.delineation_status == "ok":
                assert s.has_catchment

    def test_catchment_bboxes_are_within_devon(self, ds):
        # Devon rough bounds: lat 50.2–51.2, lon -4.7–-3.0
        for s in ds:
            if not s.has_catchment:
                continue
            west, south, east, north = s.catchment_bbox()
            assert -5.0 <= west < east <= -2.5
            assert 50.0 <= south < north <= 51.5

    def test_sample_dates_parsed(self, ds):
        for s in ds:
            assert s.date is not None
            assert isinstance(s.date, date)

    def test_result_values_are_positive(self, ds):
        for s in ds:
            if s.result is not None:
                assert s.result > 0

    def test_flow_acc_positive_when_present(self, ds):
        for s in ds:
            if s.flow_acc_at_pour_point is not None:
                assert s.flow_acc_at_pour_point > 0

    def test_summary_has_correct_length(self, ds):
        df = ds.summary()
        assert len(df) == len(ds)

    def test_to_geodataframe_round_trip(self, ds):
        gdf = ds.to_geodataframe()
        ds2 = CatchmentDataset.from_geodataframe(gdf)
        assert len(ds2) == len(ds)

    def test_filter_by_site_name(self, ds):
        result = ds.filter(site_name_contains="RIVER")
        assert len(result) > 0
        for s in result:
            assert "RIVER" in s.site_name.upper()

    def test_filter_by_result_range(self, ds):
        all_results = [s.result for s in ds if s.result is not None]
        lo, hi = min(all_results), max(all_results)
        mid = (lo + hi) / 2
        result = ds.filter(min_result=lo, max_result=mid)
        for s in result:
            if s.result is not None:
                assert s.result <= mid

    def test_with_catchments_subset(self, ds):
        with_catch = ds.with_catchments()
        assert len(with_catch) <= len(ds)
        assert all(s.has_catchment for s in with_catch)

    def test_catchment_area_reasonable(self, ds):
        for s in ds.with_catchments():
            area = s.catchment_area_km2()
            assert area is not None
            # Individual stream catchments should be between 0.01 km² and 1000 km²
            assert 0.01 < area < 1000, f"Unreasonable area for {s.site_name}: {area} km²"

    def test_snapped_pour_point_near_sample_location(self, ds):
        for s in ds.with_catchments():
            if s.snap_latitude is None:
                continue
            lat_diff = abs(s.snap_latitude - s.latitude)
            lon_diff = abs(s.snap_longitude - s.longitude)
            # Snap offset should be less than ~0.05° (~5 km)
            assert lat_diff < 0.05, f"Large snap offset for {s.site_name}: {lat_diff}°"
            assert lon_diff < 0.05, f"Large snap offset for {s.site_name}: {lon_diff}°"

    def test_catchment_bbox_contains_pour_point(self, ds):
        for s in ds.with_catchments():
            if s.snap_longitude is None:
                continue
            west, south, east, north = s.catchment_bbox()
            # The snapped pour point should lie inside (or very near) the catchment bbox
            assert west - 0.01 <= s.snap_longitude <= east + 0.01
            assert south - 0.01 <= s.snap_latitude <= north + 0.01

    def test_dataset_repr_contains_count(self, ds):
        r = repr(ds)
        assert str(len(ds)) in r
        assert "devon_water_quality_dataset.gpkg" in r
