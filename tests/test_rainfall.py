"""
tests/test_rainfall.py
----------------------
Tests for the eoflow.rainfall module.

This module tests:
- Date/time parsing (shared helper)
- NIMROD per-timestep NetCDF file discovery
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eoflow.rainfall import find_nimrod_nc_files, get_nimrod_rainfall_for_polygon, parse_datetime

# ---------------------------------------------------------------------------
# Test parse_datetime
# ---------------------------------------------------------------------------


class TestParseDatetime:
    """Tests for parse_datetime function."""

    def test_parse_datetime_with_time(self):
        """Parse YYYY-MM-DDTHH:MM format."""
        result = parse_datetime("2024-03-15T12:30")
        assert result == datetime(2024, 3, 15, 12, 30, 0, tzinfo=timezone.utc)

    def test_parse_datetime_date_only(self):
        """Parse YYYY-MM-DD format (defaults to 00:00)."""
        result = parse_datetime("2024-03-15")
        assert result == datetime(2024, 3, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_parse_datetime_is_utc_aware(self):
        """Parsed datetime should have UTC timezone."""
        result = parse_datetime("2024-03-15T12:30")
        assert result.tzinfo == timezone.utc

    def test_parse_datetime_midnight(self):
        """Parse datetime at midnight."""
        result = parse_datetime("2024-01-01T00:00")
        assert result == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_parse_datetime_invalid_format_raises(self):
        """Invalid format should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            parse_datetime("2024/03/15")

    def test_parse_datetime_empty_string_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            parse_datetime("")

    def test_parse_datetime_invalid_date_raises(self):
        """Invalid date values should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            parse_datetime("2024-13-45T12:30")

    def test_parse_datetime_with_seconds_raises(self):
        """Seconds in string should cause parse to fail (not supported)."""
        with pytest.raises(ValueError):
            parse_datetime("2024-03-15T12:30:45")


# ---------------------------------------------------------------------------
# Test find_nimrod_nc_files
# ---------------------------------------------------------------------------


class TestFindNimrodNcFiles:
    """Tests for find_nimrod_nc_files function."""

    def _touch(self, root: Path, rel: str) -> Path:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
        return f

    def test_finds_files_recursively(self, tmp_path):
        """Should find per-timestep files nested under year/day directories."""
        self._touch(tmp_path, "2021/2021/20210101/20210101_000000.nc")
        self._touch(tmp_path, "2021/2021/20210101/20210101_000500.nc")
        (tmp_path / "other.txt").touch()

        files, n_skipped = find_nimrod_nc_files(tmp_path)

        assert len(files) == 2
        assert n_skipped == 0
        assert all(f.suffix == ".nc" for f in files)

    def test_empty_directory_returns_empty(self, tmp_path):
        """Empty directory should return an empty list."""
        files, n_skipped = find_nimrod_nc_files(tmp_path)

        assert files == []
        assert n_skipped == 0

    def test_sorted_chronologically(self, tmp_path):
        """Results should be sorted (filename stems sort chronologically)."""
        self._touch(tmp_path, "20210101_001000.nc")
        self._touch(tmp_path, "20210101_000000.nc")
        self._touch(tmp_path, "20210101_000500.nc")

        files, _ = find_nimrod_nc_files(tmp_path)

        assert [f.stem for f in files] == [
            "20210101_000000",
            "20210101_000500",
            "20210101_001000",
        ]

    def test_date_range_filters_files(self, tmp_path):
        """start/end should filter out files outside the range."""
        self._touch(tmp_path, "20210101_000000.nc")
        self._touch(tmp_path, "20210102_000000.nc")
        self._touch(tmp_path, "20210103_000000.nc")

        files, n_skipped = find_nimrod_nc_files(
            tmp_path,
            start=datetime(2021, 1, 2),
            end=datetime(2021, 1, 2, 23, 59),
        )

        assert [f.stem for f in files] == ["20210102_000000"]
        assert n_skipped == 2

    def test_unparseable_stem_always_included(self, tmp_path):
        """Files whose stem doesn't match the timestamp format are kept when filtering."""
        self._touch(tmp_path, "not_a_timestamp.nc")

        files, n_skipped = find_nimrod_nc_files(
            tmp_path,
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 2),
        )

        assert [f.name for f in files] == ["not_a_timestamp.nc"]
        assert n_skipped == 0

    def test_day_stem_kept_when_day_overlaps_range(self, tmp_path):
        """A YYYYMMDD (per-day, consolidate_nimrod.sh) file overlapping the window is kept."""
        self._touch(tmp_path, "20210102.nc")

        files, n_skipped = find_nimrod_nc_files(
            tmp_path,
            # Window starts mid-day on the 2nd -- the day file still overlaps it.
            start=datetime(2021, 1, 2, 12, 0),
            end=datetime(2021, 1, 3),
        )

        assert [f.name for f in files] == ["20210102.nc"]
        assert n_skipped == 0

    def test_day_stem_skipped_when_day_outside_range(self, tmp_path):
        """A per-day file entirely outside the window is skipped."""
        self._touch(tmp_path, "20210101.nc")
        self._touch(tmp_path, "20210105.nc")

        files, n_skipped = find_nimrod_nc_files(
            tmp_path,
            start=datetime(2021, 1, 3),
            end=datetime(2021, 1, 4),
        )

        assert files == []
        assert n_skipped == 2


# ---------------------------------------------------------------------------
# Test get_nimrod_rainfall_for_polygon
# ---------------------------------------------------------------------------


class TestGetNimrodRainfallForPolygonDayFiles:
    """Regression tests for loading from a directory of per-day consolidated files.

    Per-day files can only be filtered by find_nimrod_nc_files at day
    granularity, so get_nimrod_rainfall_for_polygon must also slice to the
    exact requested window after loading -- otherwise a window starting or
    ending mid-day would silently pull in whole extra days of data.
    """

    def _write_day_file(self, nimrod_dir: Path, day: datetime, xs, ys) -> None:
        times = [day + timedelta(hours=h) for h in (0, 6, 12, 18)]
        data = np.random.rand(len(times), len(ys), len(xs)).astype("float32")
        ds = xr.Dataset(
            {
                "rainfall_rate": (
                    ("time", "projection_y_coordinate", "projection_x_coordinate"),
                    data,
                )
            },
            coords={
                "time": times,
                "projection_y_coordinate": ys,
                "projection_x_coordinate": xs,
            },
        )
        year_dir = nimrod_dir / str(day.year)
        year_dir.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(year_dir / f"{day.strftime('%Y%m%d')}.nc")

    def test_mid_day_window_excludes_out_of_range_timesteps(self, tmp_path):
        """A window starting mid-day-1 and ending before day-2 should return exactly 2 steps."""
        from pyproj import Transformer
        from shapely.geometry import box
        from shapely.ops import transform as shapely_transform

        # Small polygon over Devon, reprojected to BNG to size the synthetic grid.
        polygon = box(-3.55, 50.70, -3.45, 50.80)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        polygon_bng = shapely_transform(transformer.transform, polygon)
        x_min, y_min, x_max, y_max = polygon_bng.bounds

        xs = np.arange(x_min - 2000, x_max + 2000, 1000.0)
        ys = np.arange(y_min - 2000, y_max + 2000, 1000.0)

        nimrod_dir = tmp_path / "nimrod"
        self._write_day_file(nimrod_dir, datetime(2021, 1, 1), xs, ys)
        self._write_day_file(nimrod_dir, datetime(2021, 1, 2), xs, ys)

        da = get_nimrod_rainfall_for_polygon(
            polygon,
            "2021-01-01T12:00",
            "2021-01-01T23:00",
            nimrod_dir=nimrod_dir,
        )

        assert da.sizes["time"] == 2
        result_times = [str(t)[:16] for t in da.time.values]
        assert result_times == ["2021-01-01T12:00", "2021-01-01T18:00"]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_module_imports():
    """Verify the main NIMROD + shared functions can be imported."""
    from eoflow.rainfall import (
        find_nimrod_nc_files,
        get_nimrod_rainfall_for_polygon,
        parse_datetime,
        process_nimrod_local,
    )

    assert callable(find_nimrod_nc_files)
    assert callable(get_nimrod_rainfall_for_polygon)
    assert callable(parse_datetime)
    assert callable(process_nimrod_local)
