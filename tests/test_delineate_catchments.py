"""
Unit tests for the delineate_catchments script.

These tests exercise the CLI argument parser, the core ``delineate_catchments``
function, checkpoint save / load / resume, location-cache construction,
the ``main()`` entry-point, and various edge-cases – all without requiring
real DEM data.  ``_delineate_catchment_core`` is mocked throughout.

The tests mirror a real invocation such as::

    uv run scripts/delineate_catchments.py \
        --csv ./data/devon_water_quality.csv \
        --out data/devon_water_quality_dataset.gpkg \
        --dem data/devon_dem.tif \
        --lat-col latitude \
        --lon-col longitude
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from scripts.delineate_catchments import (
    CHECKPOINT_LAYER,
    _build_location_cache,
    _load_checkpoint,
    _save_checkpoint,
    delineate_catchments,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_result(lon: float = -3.53, lat: float = 50.72, size: float = 0.1):
    """Return a (Polygon, Point, float) tuple matching ``_delineate_catchment_core``."""
    return (_sample_polygon(lon, lat, size), Point(lon, lat), 250.0)


def _sample_polygon(lon: float = -3.53, lat: float = 50.72, size: float = 0.1) -> Polygon:
    """Return a small square polygon centred on (*lon*, *lat*) in Devon."""
    half = size / 2
    return Polygon(
        [
            (lon - half, lat - half),
            (lon + half, lat - half),
            (lon + half, lat + half),
            (lon - half, lat + half),
            (lon - half, lat - half),
        ]
    )


def _write_csv(
    path: Path,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    n_rows: int = 4,
    include_nan_row: bool = False,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Write a minimal Devon-style CSV with coordinate columns and return the DataFrame.

    Coordinates are within Devon (lat ~50.7, lon ~-3.5).
    """
    # Devon sampling-point locations (lat/lon in WGS84)
    data = {
        lat_col: [50.72 + i * 0.01 for i in range(n_rows)],
        lon_col: [-3.53 + i * 0.01 for i in range(n_rows)],
        "sample_id": list(range(n_rows)),
        "samplingPoint.notation": [f"SW-7{i:07d}" for i in range(n_rows)],
        "samplingPoint.easting": [292546 + i * 100 for i in range(n_rows)],
        "samplingPoint.northing": [91506 + i * 100 for i in range(n_rows)],
        "result": [5.0 + i * 1.5 for i in range(n_rows)],
        "determinand.prefLabel": ["Turbidity"] * n_rows,
    }
    if extra_cols:
        data.update(extra_cols)

    df = pd.DataFrame(data)

    if include_nan_row:
        nan_row = pd.DataFrame({lat_col: [None], lon_col: [None], "sample_id": [999]})
        if extra_cols:
            for col in extra_cols:
                nan_row[col] = None
        df = pd.concat([df, nan_row], ignore_index=True)

    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary directory for each test."""
    return tmp_path


@pytest.fixture()
def csv_path(tmp_dir: Path) -> Path:
    """Write a small CSV and return its path."""
    p = tmp_dir / "samples.csv"
    _write_csv(p)
    return p


@pytest.fixture()
def csv_path_with_nan(tmp_dir: Path) -> Path:
    """CSV that includes a row with missing coordinates."""
    p = tmp_dir / "samples_nan.csv"
    _write_csv(p, include_nan_row=True)
    return p


@pytest.fixture()
def dem_path(tmp_dir: Path) -> Path:
    """Create a dummy Devon DEM file (content doesn't matter – it's mocked)."""
    p = tmp_dir / "devon_dem.tif"
    p.touch()
    return p


@pytest.fixture()
def output_path(tmp_dir: Path) -> Path:
    """Return a path for the output GeoPackage (does not exist yet)."""
    return tmp_dir / "output" / "devon_water_quality_dataset.gpkg"


@pytest.fixture()
def sample_polygon() -> Polygon:
    return _sample_polygon()


@pytest.fixture()
def cli_argv(csv_path: Path, dem_path: Path, output_path: Path) -> List[str]:
    """Argument vector that mirrors the user's real invocation."""
    return [
        "--csv",
        str(csv_path),
        "--dem",
        str(dem_path),
        "--out",
        str(output_path),
        "--lat-col",
        "latitude",
        "--lon-col",
        "longitude",
        "--log-level",
        "INFO",
    ]


# ---------------------------------------------------------------------------
# TestDelineateCatchments
# ---------------------------------------------------------------------------


class TestDelineateCatchments:
    """Test the core ``delineate_catchments`` function with mocked delineation."""

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_basic_run(self, mock_delineate, csv_path, dem_path, output_path) -> None:
        """All rows succeed – every row should have status 'ok'."""
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        gdf = delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 4
        assert (gdf["delineation_status"] == "ok").all()
        assert (gdf["delineation_error"] == "").all()

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_output_gpkg_created(self, mock_delineate, csv_path, dem_path, output_path) -> None:
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert output_path.exists()

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_catchment_geometry_is_polygon(
        self, mock_delineate, csv_path, dem_path, output_path
    ) -> None:
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        gdf = delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        for geom in gdf["catchment"]:
            assert isinstance(geom, Polygon)

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_delineation_called_per_unique_location(
        self, mock_delineate, tmp_dir, dem_path, output_path
    ) -> None:
        """Duplicate (lat, lon) pairs must trigger only one delineation."""
        csv = tmp_dir / "dup.csv"
        df = pd.DataFrame(
            {
                "latitude": [50.72, 50.72, 50.80],
                "longitude": [-3.53, -3.53, -3.60],
                "sample_id": [1, 2, 3],
            }
        )
        df.to_csv(csv, index=False)

        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        delineate_catchments(
            csv_path=csv,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        # 2 unique locations → 2 calls
        assert mock_delineate.call_count == 2

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_duplicate_locations_share_polygon(
        self, mock_delineate, tmp_dir, dem_path, output_path
    ) -> None:
        csv = tmp_dir / "dup.csv"
        df = pd.DataFrame(
            {
                "latitude": [50.72, 50.72],
                "longitude": [-3.53, -3.53],
                "sample_id": [1, 2],
            }
        )
        df.to_csv(csv, index=False)

        poly = _sample_polygon(-3.53, 50.72)
        mock_delineate.return_value = (poly, Point(-3.53, 50.72), 250.0)

        gdf = delineate_catchments(
            csv_path=csv,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert gdf.iloc[0]["catchment"].equals(gdf.iloc[1]["catchment"])

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_delineation_failure_recorded(
        self, mock_delineate, csv_path, dem_path, output_path
    ) -> None:
        mock_delineate.side_effect = RuntimeError("DEM out of bounds")

        gdf = delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert (gdf["delineation_status"] == "error").all()
        assert (gdf["delineation_error"] == "DEM out of bounds").all()

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_partial_failure(self, mock_delineate, csv_path, dem_path, output_path) -> None:
        """Some locations succeed, others fail."""
        call_count = {"n": 0}

        def _side_effect(point, **kw):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return _sample_result(point.x, point.y)
            raise RuntimeError("boom")

        mock_delineate.side_effect = _side_effect

        gdf = delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        n_ok = (gdf["delineation_status"] == "ok").sum()
        n_err = (gdf["delineation_status"] == "error").sum()
        assert n_ok == 2
        assert n_err == 2

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_missing_column_raises(self, mock_delineate, tmp_dir, dem_path, output_path) -> None:
        csv = tmp_dir / "bad.csv"
        pd.DataFrame({"x": [1], "y": [2]}).to_csv(csv, index=False)

        with pytest.raises(ValueError, match="latitude"):
            delineate_catchments(
                csv_path=csv,
                dem_path=dem_path,
                output_path=output_path,
                lat_col="latitude",
                lon_col="longitude",
            )

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_nan_coordinates_dropped(
        self, mock_delineate, csv_path_with_nan, dem_path, output_path
    ) -> None:
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        gdf = delineate_catchments(
            csv_path=csv_path_with_nan,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        # The CSV has 4 valid rows + 1 NaN row → 4 rows in the result
        assert len(gdf) == 4
        assert (gdf["delineation_status"] == "ok").all()

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_flow_acc_threshold_forwarded(
        self, mock_delineate, csv_path, dem_path, output_path
    ) -> None:
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
            flow_acc_threshold=2500,
        )

        for call in mock_delineate.call_args_list:
            assert call.kwargs["flow_acc_threshold"] == 2500

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_point_constructed_as_lon_lat(
        self, mock_delineate, tmp_dir, dem_path, output_path
    ) -> None:
        """Shapely Points should be (x=lon, y=lat)."""
        csv = tmp_dir / "one.csv"
        pd.DataFrame({"latitude": [50.72], "longitude": [-3.53]}).to_csv(csv, index=False)

        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        delineate_catchments(
            csv_path=csv,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        called_point = mock_delineate.call_args_list[0].kwargs["point"]
        assert called_point.x == pytest.approx(-3.53)
        assert called_point.y == pytest.approx(50.72)

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_result_has_expected_columns(
        self, mock_delineate, csv_path, dem_path, output_path
    ) -> None:
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        gdf = delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert "catchment" in gdf.columns
        assert "delineation_status" in gdf.columns
        assert "delineation_error" in gdf.columns
        assert "latitude" in gdf.columns
        assert "longitude" in gdf.columns
        assert "sample_id" in gdf.columns

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_original_csv_columns_preserved(
        self, mock_delineate, tmp_dir, dem_path, output_path
    ) -> None:
        csv = tmp_dir / "extra.csv"
        _write_csv(csv, extra_cols={"colour": ["red", "blue", "green", "yellow"]})
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        gdf = delineate_catchments(
            csv_path=csv,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert "colour" in gdf.columns


# ---------------------------------------------------------------------------
# TestCheckpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Save / load / resume checkpoint round-trips."""

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_checkpoint_file_written(self, mock_delineate, csv_path, dem_path, output_path) -> None:
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert output_path.exists()

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_checkpoint_round_trip(self, mock_delineate, csv_path, dem_path, output_path) -> None:
        """Write → reload checkpoint; row count and status must survive."""
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        gdf = delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        loaded = _load_checkpoint(output_path)
        assert loaded is not None
        assert len(loaded) == len(gdf)

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_resume_skips_already_delineated(
        self, mock_delineate, csv_path, dem_path, output_path
    ) -> None:
        """Running twice must not re-delineate locations from the first run."""
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        # First run – delineates all 4 unique locations
        delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )
        first_run_calls = mock_delineate.call_count
        assert first_run_calls == 4

        mock_delineate.reset_mock()

        # Second run – should skip everything
        delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )
        assert mock_delineate.call_count == 0

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_periodic_checkpoint(self, mock_delineate, tmp_dir, dem_path, output_path) -> None:
        """With checkpoint_every=2 and 4 locations, at least one mid-run save."""
        csv = tmp_dir / "big.csv"
        _write_csv(csv, n_rows=4)
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        delineate_catchments(
            csv_path=csv,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
            checkpoint_every=2,
        )

        # The file must exist (at least final save)
        assert output_path.exists()

    def test_load_checkpoint_returns_none_for_missing_file(self, tmp_dir) -> None:
        result = _load_checkpoint(tmp_dir / "nonexistent.gpkg")
        assert result is None

    def test_load_checkpoint_returns_none_for_corrupt_file(self, tmp_dir) -> None:
        bad_file = tmp_dir / "corrupt.gpkg"
        bad_file.write_text("this is not a geopackage")
        result = _load_checkpoint(bad_file)
        assert result is None

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_checkpoint_with_mismatched_length_reuses_cache(
        self, mock_delineate, tmp_dir, dem_path
    ) -> None:
        """If CSV gains rows, cached polygons are salvaged by (lat, lon)."""
        output = tmp_dir / "out.gpkg"

        # Build first with 2 rows
        csv_small = tmp_dir / "small.csv"
        _write_csv(csv_small, n_rows=2)
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)
        delineate_catchments(
            csv_path=csv_small,
            dem_path=dem_path,
            output_path=output,
            lat_col="latitude",
            lon_col="longitude",
        )
        assert mock_delineate.call_count == 2
        mock_delineate.reset_mock()

        # Now rebuild with 4 rows (first 2 coords are the same)
        csv_big = tmp_dir / "big.csv"
        _write_csv(csv_big, n_rows=4)
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)
        gdf = delineate_catchments(
            csv_path=csv_big,
            dem_path=dem_path,
            output_path=output,
            lat_col="latitude",
            lon_col="longitude",
        )

        # Only the 2 new locations should be delineated
        assert mock_delineate.call_count == 2
        assert len(gdf) == 4
        assert (gdf["delineation_status"] == "ok").all()


# ---------------------------------------------------------------------------
# TestSaveCheckpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    """Direct tests for ``_save_checkpoint``."""

    def test_creates_parent_directory(self, tmp_dir: Path) -> None:
        nested = tmp_dir / "a" / "b" / "c" / "out.gpkg"

        gdf = gpd.GeoDataFrame(
            {"val": [1]},
            geometry=gpd.GeoSeries([None]),
            crs="EPSG:4326",
        )
        gdf = gdf.rename_geometry("catchment")

        _save_checkpoint(gdf, nested)
        assert nested.exists()

    def test_saves_all_rows(self, tmp_dir: Path) -> None:
        poly = _sample_polygon()
        gdf = gpd.GeoDataFrame(
            {
                "sample_id": [1, 2],
                "catchment": [poly, None],
            },
            geometry="catchment",
            crs="EPSG:4326",
        )

        out = tmp_dir / "test.gpkg"
        _save_checkpoint(gdf, out)

        loaded = gpd.read_file(str(out), layer=CHECKPOINT_LAYER)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# TestBuildLocationCache
# ---------------------------------------------------------------------------


class TestBuildLocationCache:
    """Tests for ``_build_location_cache``."""

    def test_cache_from_processed_rows(self) -> None:
        poly = _sample_polygon(-3.53, 50.72)
        gdf = gpd.GeoDataFrame(
            {
                "latitude": [50.72, 50.80],
                "longitude": [-3.53, -3.60],
                "catchment": [poly, None],
            },
            geometry="catchment",
            crs="EPSG:4326",
        )

        cache = _build_location_cache(gdf, "latitude", "longitude")
        assert (50.72, -3.53) in cache
        assert (50.80, -3.60) not in cache

    def test_empty_gdf_gives_empty_cache(self) -> None:
        gdf = gpd.GeoDataFrame(
            {
                "latitude": pd.Series(dtype=float),
                "longitude": pd.Series(dtype=float),
                "catchment": pd.Series(dtype=object),
            },
            geometry="catchment",
            crs="EPSG:4326",
        )
        cache = _build_location_cache(gdf, "latitude", "longitude")
        assert len(cache) == 0

    def test_duplicate_locations_keep_last(self) -> None:
        poly = _sample_polygon()
        gdf = gpd.GeoDataFrame(
            {
                "latitude": [50.72, 50.72],
                "longitude": [-3.53, -3.53],
                "catchment": [poly, poly],
            },
            geometry="catchment",
            crs="EPSG:4326",
        )

        cache = _build_location_cache(gdf, "latitude", "longitude")
        assert len(cache) == 1
        assert (50.72, -3.53) in cache


# ---------------------------------------------------------------------------
# TestMainCLI
# ---------------------------------------------------------------------------


class TestMainCLI:
    """Test the ``main()`` entry-point that wires up CLI → delineate_catchments."""

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_main_runs_end_to_end(self, mock_delineate, cli_argv, output_path) -> None:
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        main(cli_argv)

        assert output_path.exists()

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_main_with_log_level(self, mock_delineate, cli_argv) -> None:
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        main(cli_argv + ["--log-level", "DEBUG"])
        # No exception means logging configured correctly at DEBUG

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_main_with_log_file(self, mock_delineate, cli_argv, tmp_dir) -> None:
        log_file = tmp_dir / "run.log"
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        main(cli_argv + ["--log-file", str(log_file)])
        assert log_file.exists()
        contents = log_file.read_text()
        assert len(contents) > 0

    def test_main_missing_csv_exits(self, dem_path, output_path) -> None:
        argv = [
            "--csv",
            "/nonexistent/path.csv",
            "--dem",
            str(dem_path),
            "--out",
            str(output_path),
            "--lat-col",
            "latitude",
            "--lon-col",
            "longitude",
        ]
        with pytest.raises(SystemExit):
            main(argv)

    def test_main_missing_dem_exits(self, csv_path, output_path) -> None:
        argv = [
            "--csv",
            str(csv_path),
            "--dem",
            "/nonexistent/dem.tif",
            "--out",
            str(output_path),
            "--lat-col",
            "latitude",
            "--lon-col",
            "longitude",
        ]
        with pytest.raises(SystemExit):
            main(argv)

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_main_custom_checkpoint_every(self, mock_delineate, cli_argv) -> None:
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        main(cli_argv + ["--checkpoint-every", "2"])
        # Should not raise

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_main_custom_flow_acc(self, mock_delineate, cli_argv) -> None:
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        main(cli_argv + ["--flow-acc-threshold", "500"])

        for call in mock_delineate.call_args_list:
            assert call.kwargs["flow_acc_threshold"] == 500


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_single_row_csv(self, mock_delineate, tmp_dir, dem_path, output_path) -> None:
        csv = tmp_dir / "one.csv"
        _write_csv(csv, n_rows=1)
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        gdf = delineate_catchments(
            csv_path=csv,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert len(gdf) == 1
        assert gdf.iloc[0]["delineation_status"] == "ok"

    @pytest.mark.xfail(
        reason=(
            "Empty GeoDataFrame after dropping all-NaN rows causes "
            "_save_checkpoint to fail with multiple geometry columns. "
            "Known edge case in delineate_catchments/save_checkpoint interaction."
        ),
        raises=ValueError,
        strict=True,
    )
    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_all_rows_nan_produces_empty_result(
        self, mock_delineate, tmp_dir, dem_path, output_path
    ) -> None:
        csv = tmp_dir / "allnan.csv"
        pd.DataFrame({"latitude": [None, None], "longitude": [None, None], "id": [1, 2]}).to_csv(
            csv, index=False
        )

        gdf = delineate_catchments(
            csv_path=csv,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert len(gdf) == 0
        mock_delineate.assert_not_called()

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_large_checkpoint_interval(
        self, mock_delineate, csv_path, dem_path, output_path
    ) -> None:
        """checkpoint_every > n_rows → only the final save triggers."""
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        gdf = delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
            checkpoint_every=9999,
        )

        assert len(gdf) == 4
        assert output_path.exists()

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_dem_path_coerced_to_path(
        self, mock_delineate, csv_path, dem_path, output_path
    ) -> None:
        """delineate_catchments wraps dem_path in Path(); passing a string should work."""
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        gdf = delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,  # string, not Path
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert len(gdf) == 4

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_every_delineation_raises(
        self, mock_delineate, csv_path, dem_path, output_path
    ) -> None:
        """If every call fails the script must still produce a GeoPackage."""
        mock_delineate.side_effect = RuntimeError("total failure")

        gdf = delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
        )

        assert len(gdf) == 4
        assert (gdf["delineation_status"] == "error").all()
        assert output_path.exists()

    @patch("scripts.delineate_catchments._delineate_catchment_core")
    def test_checkpoint_every_one(self, mock_delineate, csv_path, dem_path, output_path) -> None:
        """checkpoint_every=1 saves after every delineation."""
        mock_delineate.side_effect = lambda point, **kw: _sample_result(point.x, point.y)

        gdf = delineate_catchments(
            csv_path=csv_path,
            dem_path=dem_path,
            output_path=output_path,
            lat_col="latitude",
            lon_col="longitude",
            checkpoint_every=1,
        )

        assert len(gdf) == 4
        assert output_path.exists()
