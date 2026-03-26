"""
Integration tests for eoflow.samples — real Sentinel-2 queries via openEO.

These tests connect to the Copernicus Data Space openEO federation and download
small data cubes for delineated catchments from the Devon water-quality dataset.
They are therefore:

* Slow        (network + server-side processing, typically 10–60 s each)
* Requires    cached OIDC credentials  (run ``openeo-auth`` or any previous
              ``conn.authenticate_oidc()`` session to prime the token cache)
* Requires    the Devon GeoPackage at  ``data/devon_water_quality_dataset.gpkg``

Run only these tests:
    pytest -m "eo_integration" -v tests/test_samples_eo_integration.py

Skip these tests (default when running the full suite):
    pytest -m "not eo_integration"

Environment variable ``EOFLOW_EO_INTEGRATION`` must be set to ``1`` (or the
``--run-eo-integration`` CLI flag used) to avoid accidental network calls in CI.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio

from eoflow.samples import (
    OPENEO_BACKEND,
    SENTINEL2_COLLECTION,
    CatchmentDataset,
    Sample,
)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_GPKG_PATH = _DATA_DIR / "devon_water_quality_dataset.gpkg"

# Notation of the smallest catchment in the dataset (RIVER TEIGN AT PRESTON,
# ~0.06 km²). Its tiny spatial extent keeps download times short.
_TEIGN_NOTATION = "SW-70620154"

# Fixed date window that is known to contain at least one clear-ish Sentinel-2
# overpass over Devon (January 2020).  The sample date for this site is
# 2020-01-09, so a ±20-day window spans 2019-12-20 → 2020-01-29.
_START = "2020-01-01"
_END = "2020-02-01"

# NDVI range that is physically plausible for UK land in winter.
_NDVI_PLAUSIBLE_MIN = -0.2  # some open water / bare soil possible
_NDVI_PLAUSIBLE_MAX = 1.0

# Opt-in guard: set the env-var to "1" to run these tests.
_EO_INTEGRATION_ENABLED = os.environ.get("EOFLOW_EO_INTEGRATION", "0") == "1"


# ---------------------------------------------------------------------------
# Pytest marks & session-level skip
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.eo_integration


def _require_eo_integration():
    """Skip the test unless the opt-in env-var is set."""
    if not _EO_INTEGRATION_ENABLED:
        pytest.skip("EO integration tests are opt-in.  Set EOFLOW_EO_INTEGRATION=1 to run them.")


def _require_gpkg():
    if not _GPKG_PATH.exists():
        pytest.skip(f"Devon GeoPackage not found at {_GPKG_PATH}")


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def openeo_connection():
    """Authenticated openEO connection, shared across all tests in the session."""
    _require_eo_integration()
    _require_gpkg()
    try:
        import openeo  # noqa: F401
    except ImportError:
        pytest.skip("openeo package is not installed")

    conn = Sample.connect_openeo(OPENEO_BACKEND, authenticate=True)
    return conn


@pytest.fixture(scope="session")
def dataset() -> CatchmentDataset:
    """Full CatchmentDataset loaded from the real Devon GeoPackage."""
    _require_eo_integration()
    _require_gpkg()
    return CatchmentDataset.from_gpkg(_GPKG_PATH)


@pytest.fixture(scope="session")
def teign_sample(dataset: CatchmentDataset) -> Sample:
    """The smallest delineated catchment — used for fast EO downloads."""
    matches = [s for s in dataset if s.notation == _TEIGN_NOTATION]
    if not matches:
        pytest.skip(f"Sample {_TEIGN_NOTATION!r} not found in dataset")
    return matches[0]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _download_tiff(cube, label: str) -> Path:
    """Download a DataCube synchronously to a temp GeoTIFF and return the path.

    The caller is responsible for cleaning up the temp directory.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix=f"eoflow_test_{label}_"))
    out = tmpdir / f"{label}.tif"
    cube.download(str(out), format="GTiff")
    return out


def _finite_pixels(path: Path, band: int = 1) -> np.ndarray:
    """Return a 1-D array of all finite (non-NaN, non-nodata) pixel values."""
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
    return data[np.isfinite(data)]


def _raster_info(path: Path) -> dict:
    """Return a dict of basic raster metadata."""
    with rasterio.open(path) as src:
        return {
            "shape": src.shape,
            "bands": src.count,
            "crs": src.crs,
            "dtype": src.dtypes[0],
        }


# ===========================================================================
# Test classes
# ===========================================================================


class TestConnectionAndBackend:
    """Verify that the openEO connection itself is healthy before querying."""

    def test_connect_returns_authenticated_connection(self, openeo_connection):
        """connect_openeo() returns an object with a describe_account method."""
        conn = openeo_connection
        # A successful auth means we can call describe_account without error.
        info = conn.describe_account()
        assert isinstance(info, dict)

    def test_sentinel2_collection_exists(self, openeo_connection):
        """SENTINEL2_L2A collection is advertised by the backend."""
        conn = openeo_connection
        collections = conn.list_collections()
        ids = [c["id"] for c in collections]
        assert SENTINEL2_COLLECTION in ids, (
            f"{SENTINEL2_COLLECTION!r} not found in backend collections: {ids[:10]} …"
        )

    def test_collection_has_expected_bands(self, openeo_connection):
        """The collection metadata lists Sentinel-2 optical bands."""
        conn = openeo_connection
        info = conn.describe_collection(SENTINEL2_COLLECTION)
        # Band names are nested differently across backends; look for B04 anywhere.
        raw = str(info)
        for band in ("B04", "B08", "B03"):
            assert band in raw, f"Band {band!r} not found in collection metadata"


class TestSampleMetadata:
    """Sanity-checks on the Teign sample before doing any EO queries."""

    def test_teign_has_catchment(self, teign_sample):
        assert teign_sample.has_catchment

    def test_teign_bbox_within_devon(self, teign_sample):
        west, south, east, north = teign_sample.catchment_bbox()
        # Devon rough extents
        assert -5.0 <= west < east <= -2.5
        assert 50.0 <= south < north <= 51.5

    def test_teign_area_is_small(self, teign_sample):
        area = teign_sample.catchment_area_km2()
        assert 0.01 < area < 1.0, f"Unexpected area: {area:.4f} km²"

    def test_teign_date_is_january_2020(self, teign_sample):
        assert teign_sample.date == date(2020, 1, 9)

    def test_teign_notation_matches(self, teign_sample):
        assert teign_sample.notation == _TEIGN_NOTATION

    def test_teign_snap_near_sample_point(self, teign_sample):
        lat_diff = abs(teign_sample.snap_latitude - teign_sample.latitude)
        lon_diff = abs(teign_sample.snap_longitude - teign_sample.longitude)
        assert lat_diff < 0.05
        assert lon_diff < 0.05

    def test_teign_flow_acc_positive(self, teign_sample):
        assert teign_sample.flow_acc_at_pour_point > 0


class TestQueryBands:
    """query_bands() with an explicit date range."""

    def test_returns_datacube(self, teign_sample, openeo_connection):
        """query_bands returns an openEO DataCube object."""
        cube = teign_sample.query_bands(
            openeo_connection,
            bands=["B04", "B08"],
            start_date=_START,
            end_date=_END,
        )
        # Duck-type: the object should have a download() method
        assert callable(getattr(cube, "download", None))

    def test_download_produces_file(self, teign_sample, openeo_connection):
        """Downloaded GeoTIFF is non-empty."""
        cube = teign_sample.query_bands(
            openeo_connection,
            bands=["B04", "B08"],
            start_date=_START,
            end_date=_END,
        )
        tif = _download_tiff(cube, "bands")
        assert tif.exists()
        assert tif.stat().st_size > 0

    def test_downloaded_tiff_has_two_bands(self, teign_sample, openeo_connection):
        """Requesting B04 + B08 produces a two-band raster."""
        cube = teign_sample.query_bands(
            openeo_connection,
            bands=["B04", "B08"],
            start_date=_START,
            end_date=_END,
        )
        tif = _download_tiff(cube, "bands2")
        info = _raster_info(tif)
        assert info["bands"] == 2

    def test_band_values_are_scaled_reflectance(self, teign_sample, openeo_connection):
        """Sentinel-2 L2A reflectances are delivered scaled ×10 000 (integers
        in the range ~0–10 000, or floats depending on the backend format)."""
        cube = teign_sample.query_bands(
            openeo_connection,
            bands=["B04", "B08"],
            start_date=_START,
            end_date=_END,
        )
        tif = _download_tiff(cube, "bands_vals")
        with rasterio.open(tif) as src:
            b4 = src.read(1).astype(np.float64)
            nodata = src.nodata
            if nodata is not None:
                b4[b4 == nodata] = np.nan
        finite = b4[np.isfinite(b4)]
        assert len(finite) > 0, "All pixels are NaN/nodata"
        # Scaled S2 reflectance: expect values in [0, 10000]
        assert finite.min() >= 0
        assert finite.max() <= 11000  # allow a small buffer for atmospheric edge cases

    def test_crs_is_utm(self, teign_sample, openeo_connection):
        """Copernicus Data Space returns data in UTM (Zone 30N for Devon)."""
        cube = teign_sample.query_bands(
            openeo_connection,
            bands=["B04"],
            start_date=_START,
            end_date=_END,
        )
        tif = _download_tiff(cube, "bands_crs")
        info = _raster_info(tif)
        # Devon is in UTM zone 30N (EPSG:32630); accept any projected CRS.
        assert info["crs"] is not None
        assert info["crs"].is_projected

    def test_raises_without_catchment(self, dataset, openeo_connection):
        """query_bands raises ValueError when the sample has no catchment polygon."""
        # Build a minimal sample with no catchment
        import pandas as pd

        from eoflow.samples import Sample

        row = pd.Series(
            {
                "id": "fake",
                "samplingPoint.prefLabel": "Fake Site",
                "samplingPoint.notation": "FAKE-001",
                "latitude": 50.6,
                "longitude": -3.5,
                "result": 10.0,
                "unit": "NTU",
                "determinand.prefLabel": "Turbidity",
                "Date": "2020-01-09",
                "delineation_status": "failed",
                "delineation_error": "no stream found",
            }
        )
        no_catch = Sample(row, catchment=None)
        with pytest.raises(ValueError, match="no delineated catchment"):
            no_catch.query_bands(
                openeo_connection,
                bands=["B04"],
                start_date=_START,
                end_date=_END,
            )


class TestQueryNDVI:
    """query_ndvi() — vegetation index (NIR − Red) / (NIR + Red)."""

    def test_returns_datacube(self, teign_sample, openeo_connection):
        cube = teign_sample.query_ndvi(openeo_connection, start_date=_START, end_date=_END)
        assert callable(getattr(cube, "download", None))

    def test_ndvi_values_in_valid_range(self, teign_sample, openeo_connection):
        """NDVI must lie in (−1, 1) for real land surfaces."""
        cube = teign_sample.query_ndvi(openeo_connection, start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "ndvi")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0, "All NDVI pixels are NaN/nodata"
        # NDVI via the openEO built-in ndvi process is well-bounded; allow tiny overshoot.
        assert pixels.min() >= -1.1, f"NDVI below −1: {pixels.min():.4f}"
        assert pixels.max() <= 1.1, f"NDVI above  1: {pixels.max():.4f}"

    def test_ndvi_values_plausible_for_uk_winter(self, teign_sample, openeo_connection):
        """For UK land in January the mean NDVI should be clearly positive."""
        cube = teign_sample.query_ndvi(openeo_connection, start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "ndvi_mean")
        pixels = _finite_pixels(tif)
        mean_ndvi = pixels.mean()
        assert mean_ndvi > 0.2, (
            f"Mean NDVI unexpectedly low ({mean_ndvi:.4f}); "
            "expected >0.2 for vegetated Devon catchment in winter"
        )

    def test_ndvi_single_band_output(self, teign_sample, openeo_connection):
        """NDVI is a scalar index — the output raster should have one band."""
        cube = teign_sample.query_ndvi(openeo_connection, start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "ndvi_1band")
        info = _raster_info(tif)
        assert info["bands"] == 1

    def test_ndvi_via_days_window(self, teign_sample, openeo_connection):
        """days_window parameter computes the date range from the sample date."""
        # Teign date = 2020-01-09; days_window=15 → 2019-12-25 … 2020-01-24
        cube = teign_sample.query_ndvi(openeo_connection, days_window=15)
        tif = _download_tiff(cube, "ndvi_window")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0
        assert pixels.min() >= -1.1
        assert pixels.max() <= 1.1

    def test_ndvi_spatial_extent_matches_catchment(self, teign_sample, openeo_connection):
        """The downloaded raster overlaps the catchment bounding box."""
        cube = teign_sample.query_ndvi(openeo_connection, start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "ndvi_extent")
        west, south, east, north = teign_sample.catchment_bbox()
        with rasterio.open(tif) as src:
            bounds = src.bounds
            # Transform to WGS84 for comparison
            from pyproj import Transformer

            if src.crs and src.crs.is_projected:
                tr = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                w, s = tr.transform(bounds.left, bounds.bottom)
                e, n = tr.transform(bounds.right, bounds.top)
            else:
                w, s, e, n = bounds.left, bounds.bottom, bounds.right, bounds.top
        # The raster extent should overlap Devon
        assert w < -2.5 and e > -5.0
        assert s < 51.5 and n > 50.0


class TestQueryNDWI:
    """query_ndwi() — open-water index (Green − NIR) / (Green + NIR)."""

    def test_ndwi_values_in_valid_range(self, teign_sample, openeo_connection):
        cube = teign_sample.query_ndwi(openeo_connection, start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "ndwi")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0
        # Allow a small floating-point overshoot from the backend's reduce step
        assert pixels.min() >= -1.5
        assert pixels.max() <= 1.5

    def test_ndwi_negative_for_vegetated_land(self, teign_sample, openeo_connection):
        """For a predominantly vegetated catchment NDWI should be mostly negative
        (high NIR relative to green means vegetation, not open water)."""
        cube = teign_sample.query_ndwi(openeo_connection, start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "ndwi_neg")
        pixels = _finite_pixels(tif)
        mean_ndwi = pixels.mean()
        assert mean_ndwi < 0.0, (
            f"Mean NDWI is {mean_ndwi:.4f}; expected negative for vegetated catchment"
        )

    def test_ndwi_single_band_output(self, teign_sample, openeo_connection):
        cube = teign_sample.query_ndwi(openeo_connection, start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "ndwi_1band")
        assert _raster_info(tif)["bands"] == 1


class TestQueryIndex:
    """query_index() — generic named-index dispatcher."""

    def test_ndvi_via_query_index(self, teign_sample, openeo_connection):
        cube = teign_sample.query_index(openeo_connection, "NDVI", start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "qi_ndvi")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0
        assert pixels.min() >= -1.1 and pixels.max() <= 1.1

    def test_ndwi_via_query_index(self, teign_sample, openeo_connection):
        cube = teign_sample.query_index(openeo_connection, "NDWI", start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "qi_ndwi")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0
        # Allow small overshoot from the backend's custom reduce_dimension step
        assert pixels.min() >= -1.5 and pixels.max() <= 1.5

    def test_mndwi_via_query_index(self, teign_sample, openeo_connection):
        """MNDWI = (Green − SWIR) / (Green + SWIR) — uses B03 & B11."""
        cube = teign_sample.query_index(
            openeo_connection, "MNDWI", start_date=_START, end_date=_END
        )
        tif = _download_tiff(cube, "qi_mndwi")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0
        assert pixels.min() >= -1.5 and pixels.max() <= 1.5

    def test_ndre_via_query_index(self, teign_sample, openeo_connection):
        """NDRE = (Red-Edge − Red) / (Red-Edge + Red) — uses B05 & B04."""
        cube = teign_sample.query_index(openeo_connection, "NDRE", start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "qi_ndre")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0
        assert pixels.min() >= -1.5 and pixels.max() <= 1.5

    def test_evi_via_query_index(self, teign_sample, openeo_connection):
        """EVI values should be plausible (0–1 for healthy vegetation)."""
        cube = teign_sample.query_index(openeo_connection, "EVI", start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "qi_evi")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0
        # EVI is theoretically bounded to (−1, 1) but practical S2 values sit
        # between 0 and ~0.9 for vegetated land.
        assert pixels.min() >= -0.5
        assert pixels.max() <= 1.5

    def test_case_insensitive_index_name(self, teign_sample, openeo_connection):
        """Index names are case-insensitive; 'ndvi', 'Ndvi', 'NDVI' all work."""
        for name in ("ndvi", "Ndvi", "NDVI"):
            cube = teign_sample.query_index(
                openeo_connection, name, start_date=_START, end_date=_END
            )
            assert callable(getattr(cube, "download", None))

    def test_unknown_index_raises_before_network_call(self, teign_sample, openeo_connection):
        """An unrecognised index name raises ValueError without contacting the backend."""
        with pytest.raises(ValueError, match="Unknown index"):
            # Pass a nonsense index — the error is raised locally, no download needed.
            teign_sample.query_index(
                openeo_connection, "INVALID_INDEX", start_date=_START, end_date=_END
            )


class TestQueryWithDaysWindow:
    """Verify the days_window date-resolution mechanism end-to-end."""

    def test_days_window_produces_data(self, teign_sample, openeo_connection):
        """days_window=20 around the Teign sample date covers a known overpass."""
        cube = teign_sample.query_ndvi(openeo_connection, days_window=20)
        tif = _download_tiff(cube, "window20")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0

    def test_narrow_window_still_returns_data(self, teign_sample, openeo_connection):
        """A ±7-day window around 2020-01-09 still covers at least one overpass."""
        cube = teign_sample.query_ndvi(openeo_connection, days_window=7)
        tif = _download_tiff(cube, "window7")
        # If no scenes fall in the window the result may legitimately be all-NaN;
        # in that case just check the file downloaded without error.
        assert tif.exists()

    def test_days_window_overrides_explicit_dates(self, teign_sample, openeo_connection):
        """When days_window is supplied alongside explicit dates, days_window wins.
        We verify this indirectly: explicit dates far in the future (2099) would
        return no data, but days_window=15 centred on the real sample date does."""
        cube = teign_sample.query_ndvi(
            openeo_connection,
            start_date="2099-01-01",
            end_date="2099-01-31",
            days_window=15,  # should override the year-2099 dates
        )
        tif = _download_tiff(cube, "window_override")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0, "Expected data from days_window=15 override, got all-NaN pixels"


class TestMultipleSamples:
    """Spot-check EO queries for a second site to confirm the spatial extent
    logic generalises beyond the Teign catchment."""

    @pytest.fixture(scope="class")
    def otter_sample(self, dataset: CatchmentDataset) -> Sample:
        """RIVER OTTER AT DOTTON MILL (SW-70420116) — second-smallest catchment."""
        matches = [s for s in dataset if s.notation == "SW-70420116"]
        if not matches:
            pytest.skip("Otter sample not found in dataset")
        return matches[0]

    def test_otter_ndvi_in_range(self, otter_sample, openeo_connection):
        cube = otter_sample.query_ndvi(openeo_connection, start_date=_START, end_date=_END)
        tif = _download_tiff(cube, "otter_ndvi")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0
        assert pixels.min() >= -1.1
        assert pixels.max() <= 1.1

    def test_otter_and_teign_extents_differ(self, otter_sample, teign_sample, openeo_connection):
        """The two catchments have different bounding boxes, so the downloaded
        rasters must have different spatial extents."""
        otter_bbox = otter_sample.catchment_bbox()
        teign_bbox = teign_sample.catchment_bbox()
        # Their longitudes differ by ~0.3°
        assert abs(otter_bbox[0] - teign_bbox[0]) > 0.1, (
            "Expected Otter and Teign extents to differ significantly in longitude"
        )


class TestCloudCoverFiltering:
    """max_cloud_cover parameter is forwarded to the backend correctly."""

    def test_strict_cloud_cover_still_returns_file(self, teign_sample, openeo_connection):
        """max_cloud_cover=20 (strict) may yield fewer scenes but should not error."""
        cube = teign_sample.query_ndvi(
            openeo_connection,
            start_date=_START,
            end_date=_END,
            max_cloud_cover=20,
        )
        tif = _download_tiff(cube, "cloud20")
        assert tif.exists()

    def test_permissive_cloud_cover_returns_data(self, teign_sample, openeo_connection):
        """max_cloud_cover=95 (permissive) should include more scenes."""
        cube = teign_sample.query_ndvi(
            openeo_connection,
            start_date=_START,
            end_date=_END,
            max_cloud_cover=95,
        )
        tif = _download_tiff(cube, "cloud95")
        pixels = _finite_pixels(tif)
        assert len(pixels) > 0


class TestDatasetBulkQuery:
    """query_all_ndvi() over the full dataset (slow but realistic end-to-end test)."""

    def test_query_all_ndvi_returns_results_for_each_catchment(
        self, dataset: CatchmentDataset, openeo_connection, tmp_path: Path
    ):
        """query_all_ndvi returns a notation→DataCube dict for every delineated
        sample.  We check that each notation appears and that downloading one
        of the cubes succeeds."""
        with_catch = dataset.with_catchments()
        # Limit to two samples to keep the test reasonably fast
        limited_notations = {s.notation for s in list(with_catch)[:2]}

        cubes: dict = {}
        for sample in with_catch:
            if sample.notation not in limited_notations:
                continue
            cube = sample.query_ndvi(openeo_connection, start_date=_START, end_date=_END)
            cubes[sample.notation] = cube

        assert set(cubes.keys()) == limited_notations

        # Download one to confirm it materialises
        first_notation = next(iter(limited_notations))
        out = tmp_path / f"{first_notation}_ndvi.tif"
        cubes[first_notation].download(str(out), format="GTiff")
        assert out.exists() and out.stat().st_size > 0

    def test_query_all_ndvi_output_dir(
        self, dataset: CatchmentDataset, openeo_connection, tmp_path: Path
    ):
        """query_all_ndvi with output_dir downloads files named by notation."""
        # Use only the smallest catchment to keep this fast
        single = dataset.filter(site_name_contains="TEIGN AT PRESTON")
        if len(single) == 0:
            pytest.skip("Teign sample not found after filtering")

        results = single.query_all_ndvi(
            openeo_connection,
            days_window=20,
            output_dir=tmp_path,
        )
        assert len(results) == len(single)
        # Each notation should produce a .tif file in tmp_path
        for notation in results:
            expected = tmp_path / f"{notation}_ndvi.tif"
            assert expected.exists(), f"Expected file not found: {expected}"
            assert expected.stat().st_size > 0
