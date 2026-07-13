"""
tests/test_rainfall.py
----------------------
Tests for the eoflow.rainfall module.

This module tests:
- Date/time parsing (shared helper)
- NIMROD per-timestep NetCDF file discovery
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from eoflow.rainfall import find_nimrod_nc_files, parse_datetime

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
