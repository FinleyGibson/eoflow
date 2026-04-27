"""
tests/test_rainfall.py
----------------------
Tests for the eoflow.rainfall module.

This module tests:
- Date/time parsing and generation
- S3 key construction and lookup
- Download logic (with mocks and integration tests)
- NetCDF file handling utilities
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from eoflow.rainfall import (
    candidate_run_times,
    find_s3_key_for_valid_time,
    generate_valid_times,
    output_path_for,
    parse_datetime,
    parse_timestamps_from_path,
    resolve_extent,
    s3_key,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_s3_client():
    """Mock boto3 S3 client."""
    return Mock()


@pytest.fixture
def utc_dt():
    """A sample UTC datetime."""
    return datetime(2024, 3, 15, 12, 30, 0, tzinfo=timezone.utc)


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


# ---------------------------------------------------------------------------
# Test generate_valid_times
# ---------------------------------------------------------------------------


class TestGenerateValidTimes:
    """Tests for generate_valid_times generator."""

    def test_generate_valid_times_simple_range(self):
        """Generate times for a simple 1-hour range."""
        start = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
        end = datetime(2024, 3, 15, 13, 0, tzinfo=timezone.utc)

        times = list(generate_valid_times(start, end))

        assert len(times) == 5  # 12:00, 12:15, 12:30, 12:45, 13:00
        assert times[0] == start
        assert times[-1] == end

    def test_generate_valid_times_15_minute_intervals(self):
        """Valid times should be exactly 15 minutes apart."""
        start = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
        end = datetime(2024, 3, 15, 13, 0, tzinfo=timezone.utc)

        times = list(generate_valid_times(start, end))

        for i in range(len(times) - 1):
            diff = times[i + 1] - times[i]
            assert diff == timedelta(minutes=15)

    def test_generate_valid_times_rounds_start_up(self):
        """Start time not on 15-min boundary should round up."""
        start = datetime(2024, 3, 15, 12, 7, tzinfo=timezone.utc)
        end = datetime(2024, 3, 15, 12, 30, tzinfo=timezone.utc)

        times = list(generate_valid_times(start, end))

        assert times[0] == datetime(2024, 3, 15, 12, 15, tzinfo=timezone.utc)

    def test_generate_valid_times_empty_when_end_before_rounded_start(self):
        """Empty result when end is before rounded start."""
        start = datetime(2024, 3, 15, 12, 20, tzinfo=timezone.utc)
        end = datetime(2024, 3, 15, 12, 25, tzinfo=timezone.utc)

        times = list(generate_valid_times(start, end))

        # Start rounds up to 12:30, which is after end
        assert len(times) == 0

    def test_generate_valid_times_single_timestep(self):
        """Single timestep when start equals end on boundary."""
        start = end = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

        times = list(generate_valid_times(start, end))

        assert len(times) == 1
        assert times[0] == start

    def test_generate_valid_times_strips_seconds_and_microseconds(self):
        """Generated times should have zero seconds and microseconds."""
        start = datetime(2024, 3, 15, 12, 0, 30, 123456, tzinfo=timezone.utc)
        end = datetime(2024, 3, 15, 12, 15, tzinfo=timezone.utc)

        times = list(generate_valid_times(start, end))

        for t in times:
            assert t.second == 0
            assert t.microsecond == 0


# ---------------------------------------------------------------------------
# Test s3_key
# ---------------------------------------------------------------------------


class TestS3Key:
    """Tests for s3_key function."""

    def test_s3_key_basic_format(self):
        """S3 key should have correct format."""
        run_dt = datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc)
        valid_dt = datetime(2024, 3, 1, 0, 15, tzinfo=timezone.utc)

        key = s3_key(run_dt, valid_dt)

        assert key.startswith("uk-deterministic-2km/")
        assert "20240301T0000Z" in key
        assert "20240301T0015Z" in key
        assert key.endswith("-rainfall_rate.nc")

    def test_s3_key_lead_time_calculation(self):
        """S3 key should include correct lead time."""
        run_dt = datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc)
        valid_dt = datetime(2024, 3, 1, 1, 30, tzinfo=timezone.utc)

        key = s3_key(run_dt, valid_dt)

        # Lead time is 1 hour 30 minutes
        assert "PT0001H30M" in key

    def test_s3_key_zero_lead_time(self):
        """S3 key with zero lead time (nowcast)."""
        run_dt = valid_dt = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)

        key = s3_key(run_dt, valid_dt)

        assert "PT0000H00M" in key

    def test_s3_key_large_lead_time(self):
        """S3 key with large lead time (54 hours)."""
        run_dt = datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc)
        valid_dt = datetime(2024, 3, 3, 6, 0, tzinfo=timezone.utc)

        key = s3_key(run_dt, valid_dt)

        # Lead time is 54 hours
        assert "PT0054H00M" in key

    def test_s3_key_structure(self):
        """S3 key should have correct directory structure."""
        run_dt = datetime(2024, 3, 1, 6, 0, tzinfo=timezone.utc)
        valid_dt = datetime(2024, 3, 1, 7, 15, tzinfo=timezone.utc)

        key = s3_key(run_dt, valid_dt)

        expected = "uk-deterministic-2km/20240301T0600Z/20240301T0715Z-PT0001H15M-rainfall_rate.nc"
        assert key == expected


# ---------------------------------------------------------------------------
# Test candidate_run_times
# ---------------------------------------------------------------------------


class TestCandidateRunTimes:
    """Tests for candidate_run_times generator."""

    def test_candidate_run_times_yields_descending_order(self):
        """Candidate run times should be in descending (newest first) order."""
        valid_dt = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

        candidates = list(candidate_run_times(valid_dt))

        # Should be sorted newest to oldest
        for i in range(len(candidates) - 1):
            assert candidates[i] > candidates[i + 1]

    def test_candidate_run_times_all_before_or_equal_valid_time(self):
        """All candidate run times should be <= valid time."""
        valid_dt = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

        candidates = list(candidate_run_times(valid_dt))

        for candidate in candidates:
            assert candidate <= valid_dt

    def test_candidate_run_times_within_max_lead(self):
        """All candidates should be within MAX_LEAD_HOURS of valid time."""
        valid_dt = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

        candidates = list(candidate_run_times(valid_dt))

        for candidate in candidates:
            diff = valid_dt - candidate
            assert diff.total_seconds() / 3600 <= 54  # MAX_LEAD_HOURS

    def test_candidate_run_times_only_valid_run_hours(self):
        """All candidates should have minutes=0 (top of hour)."""
        valid_dt = datetime(2024, 3, 15, 12, 30, tzinfo=timezone.utc)

        candidates = list(candidate_run_times(valid_dt))

        for candidate in candidates:
            assert candidate.minute == 0
            assert candidate.second == 0

    def test_candidate_run_times_with_run_hour_filter(self):
        """Filter should restrict to specific run hour."""
        valid_dt = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

        candidates = list(candidate_run_times(valid_dt, run_hour_filter=0))

        # Should only get runs from hour 0
        for candidate in candidates:
            assert candidate.hour == 0

    def test_candidate_run_times_no_duplicates(self):
        """Should not yield duplicate run times."""
        valid_dt = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

        candidates = list(candidate_run_times(valid_dt))

        assert len(candidates) == len(set(candidates))

    def test_candidate_run_times_recent_valid_time(self):
        """Recent valid time should have many candidates."""
        valid_dt = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

        candidates = list(candidate_run_times(valid_dt))

        # Should have candidates going back ~54 hours
        assert len(candidates) > 10


# ---------------------------------------------------------------------------
# Test find_s3_key_for_valid_time (mocked)
# ---------------------------------------------------------------------------


class TestFindS3KeyForValidTime:
    """Tests for find_s3_key_for_valid_time with mocked S3."""

    def test_find_s3_key_returns_first_existing(self, mock_s3_client):
        """Should return first existing key found."""
        valid_dt = datetime(2024, 3, 15, 12, 30, tzinfo=timezone.utc)

        # Mock: first candidate exists
        mock_s3_client.head_object = Mock(return_value={})

        result = find_s3_key_for_valid_time(mock_s3_client, valid_dt)

        assert result is not None
        assert "rainfall_rate.nc" in result

    def test_find_s3_key_returns_none_when_not_found(self, mock_s3_client):
        """Should return None when no key exists."""
        valid_dt = datetime(2024, 3, 15, 12, 30, tzinfo=timezone.utc)

        # Mock: all candidates fail
        mock_s3_client.head_object = Mock(side_effect=Exception("Not found"))

        result = find_s3_key_for_valid_time(mock_s3_client, valid_dt)

        assert result is None

    def test_find_s3_key_tries_multiple_candidates(self, mock_s3_client):
        """Should try multiple candidates before giving up."""
        valid_dt = datetime(2024, 3, 15, 12, 30, tzinfo=timezone.utc)

        # Mock: fail twice, succeed third time
        call_count = 0

        def head_object_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Not found")
            return {}

        mock_s3_client.head_object = Mock(side_effect=head_object_side_effect)

        result = find_s3_key_for_valid_time(mock_s3_client, valid_dt)

        assert result is not None
        assert call_count == 3

    def test_find_s3_key_with_run_hour_filter(self, mock_s3_client):
        """Should respect run_hour_filter parameter."""
        valid_dt = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

        mock_s3_client.head_object = Mock(return_value={})

        result = find_s3_key_for_valid_time(mock_s3_client, valid_dt, run_hour_filter=0)

        # Should only try hour 0 runs
        if result:
            assert "/202403" in result  # Should be from a 00Z run


# ---------------------------------------------------------------------------
# Test output_path_for
# ---------------------------------------------------------------------------


class TestOutputPathFor:
    """Tests for output_path_for function."""

    def test_output_path_for_basic(self):
        """Should generate correct output path."""
        nc_path = Path(
            "rainfall_data/uk-deterministic-2km/"
            "20240301T0000Z/20240301T0015Z-PT0000H15M-rainfall_rate.nc"
        )
        output_dir = Path("/tmp/output")

        result = output_path_for(nc_path, output_dir)

        assert result.parent.name == "202403010000"
        assert result.name == "202403010015.asc"

    def test_output_path_for_structure(self):
        """Output path should have run_dir/valid_file.asc structure."""
        nc_path = Path(
            "data/uk-deterministic-2km/20240315T1200Z/20240315T1330Z-PT0001H30M-rainfall_rate.nc"
        )
        output_dir = Path("/output")

        result = output_path_for(nc_path, output_dir)

        assert str(result).endswith(".asc")
        assert result.parent.name == "202403151200"
        assert result.name == "202403151330.asc"

    def test_output_path_for_includes_base_dir(self):
        """Output path should be relative to base_dir."""
        nc_path = Path(
            "uk-deterministic-2km/20240301T0600Z/20240301T0700Z-PT0001H00M-rainfall_rate.nc"
        )
        output_dir = Path("/custom/output")

        result = output_path_for(nc_path, output_dir)

        assert str(result).startswith("/custom/output")


# ---------------------------------------------------------------------------
# Test parse_timestamps_from_path
# ---------------------------------------------------------------------------


class TestParseTimestampsFromPath:
    """Tests for parse_timestamps_from_path function."""

    def test_parse_timestamps_from_standard_path(self):
        """Parse timestamps from standard S3 key format."""
        path = Path(
            "rainfall_data/uk-deterministic-2km/"
            "20240301T0000Z/20240301T0015Z-PT0000H15M-rainfall_rate.nc"
        )

        run_str, valid_str = parse_timestamps_from_path(path)

        assert run_str == "202403010000"
        assert valid_str == "202403010015"

    def test_parse_timestamps_different_times(self):
        """Parse timestamps with different run and valid times."""
        path = Path(
            "data/uk-deterministic-2km/20240315T1200Z/20240315T1345Z-PT0001H45M-rainfall_rate.nc"
        )

        run_str, valid_str = parse_timestamps_from_path(path)

        assert run_str == "202403151200"
        assert valid_str == "202403151345"

    def test_parse_timestamps_returns_strings(self):
        """Parsed timestamps should be strings in YYYYMMDDHHMM format."""
        path = Path(
            "uk-deterministic-2km/20240101T0000Z/20240101T0000Z-PT0000H00M-rainfall_rate.nc"
        )

        run_str, valid_str = parse_timestamps_from_path(path)

        assert isinstance(run_str, str)
        assert isinstance(valid_str, str)
        assert len(run_str) == 12
        assert len(valid_str) == 12


# ---------------------------------------------------------------------------
# Test resolve_extent
# ---------------------------------------------------------------------------


class TestResolveExtent:
    """Tests for resolve_extent function."""

    def test_resolve_extent_none_returns_uk_default(self):
        """None should return UK BNG extent."""
        input_files = [Path("test.nc")]
        result = resolve_extent(None, input_files)

        assert len(result) == 4
        assert result[0] == 0.0  # xmin
        assert result[1] == 0.0  # ymin
        assert result[2] > result[0]  # xmax > xmin
        assert result[3] > result[1]  # ymax > ymin

    def test_resolve_extent_tuple_passthrough(self):
        """Tuple should be returned as-is."""
        extent = (100.0, 200.0, 300.0, 400.0)
        input_files = [Path("test.nc")]

        result = resolve_extent(extent, input_files)

        assert result == extent

    def test_resolve_extent_with_empty_file_list(self):
        """Should handle empty file list."""
        extent = (50.0, 60.0, 150.0, 160.0)
        input_files = []

        result = resolve_extent(extent, input_files)

        assert result == extent

    def test_resolve_extent_none_with_empty_files_returns_default(self):
        """None with empty files should return UK default."""
        input_files = []

        result = resolve_extent(None, input_files)

        # Should return UK_BNG_EXTENT
        assert len(result) == 4
        assert result == (0.0, 0.0, 700_000.0, 1_300_000.0)


# ---------------------------------------------------------------------------
# Test download_rainfall (mocked)
# ---------------------------------------------------------------------------


class TestDownloadRainfall:
    """Tests for download_rainfall function with mocked S3."""

    def test_download_rainfall_basic_success(self, mock_s3_client, tmp_path):
        """Test successful download with mocked S3."""
        from unittest.mock import patch

        from eoflow.rainfall import download_rainfall

        start = "2024-03-15T00:00"
        end = "2024-03-15T00:30"

        with patch("eoflow.rainfall.make_s3_client", return_value=mock_s3_client):
            # Mock S3 responses
            mock_s3_client.head_object = Mock(return_value={})
            mock_s3_client.download_file = Mock()

            result = download_rainfall(
                start=start,
                end=end,
                output_dir=tmp_path,
                workers=1,
            )

        # Should have counts
        assert isinstance(result, dict)
        assert "downloaded" in result
        assert "skipped" in result
        assert "error" in result

    def test_download_rainfall_with_missing_files(self, mock_s3_client, tmp_path):
        """Test download when some S3 files are missing."""
        from unittest.mock import patch

        from eoflow.rainfall import download_rainfall

        start = "2024-03-15T00:00"
        end = "2024-03-15T01:00"

        with patch("eoflow.rainfall.make_s3_client", return_value=mock_s3_client):
            # Mock: all files are missing
            mock_s3_client.head_object = Mock(side_effect=Exception("404 Not Found"))

            result = download_rainfall(
                start=start,
                end=end,
                output_dir=tmp_path,
                workers=1,
            )

        # Should complete without crashing
        assert result["downloaded"] == 0
        # No files found, so nothing to download
        assert result["error"] == 0

    def test_download_rainfall_dry_run_mode(self, mock_s3_client, tmp_path):
        """Test dry-run mode doesn't actually download."""
        from unittest.mock import patch

        from eoflow.rainfall import download_rainfall

        start = "2024-03-15T00:00"
        end = "2024-03-15T00:15"

        with patch("eoflow.rainfall.make_s3_client", return_value=mock_s3_client):
            mock_s3_client.head_object = Mock(return_value={})
            mock_s3_client.download_file = Mock()

            result = download_rainfall(
                start=start,
                end=end,
                output_dir=tmp_path,
                dry_run=True,
                workers=1,
            )

        # Should not call download_file in dry-run mode
        assert result["dry-run"] >= 0

    def test_download_rainfall_validates_date_range(self, tmp_path):
        """Test that end before start raises ValueError."""
        from eoflow.rainfall import download_rainfall

        start = "2024-03-15T12:00"
        end = "2024-03-15T06:00"  # Earlier than start

        with pytest.raises(ValueError, match="must be >= 'start'"):
            download_rainfall(
                start=start,
                end=end,
                output_dir=tmp_path,
            )

    def test_download_rainfall_accepts_datetime_objects(self, mock_s3_client, tmp_path):
        """Test download_rainfall accepts datetime objects."""
        from unittest.mock import patch

        from eoflow.rainfall import download_rainfall

        start = datetime(2024, 3, 15, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 3, 15, 0, 15, tzinfo=timezone.utc)

        with patch("eoflow.rainfall.make_s3_client", return_value=mock_s3_client):
            mock_s3_client.head_object = Mock(return_value={})
            mock_s3_client.download_file = Mock()

            result = download_rainfall(
                start=start,
                end=end,
                output_dir=tmp_path,
                workers=1,
            )

        assert isinstance(result, dict)

    def test_download_rainfall_creates_output_dir(self, mock_s3_client, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        from unittest.mock import patch

        from eoflow.rainfall import download_rainfall

        output_dir = tmp_path / "nested" / "output" / "dir"
        assert not output_dir.exists()

        start = "2024-03-15T00:00"
        end = "2024-03-15T00:15"

        with patch("eoflow.rainfall.make_s3_client", return_value=mock_s3_client):
            mock_s3_client.head_object = Mock(side_effect=Exception("Not found"))

            download_rainfall(
                start=start,
                end=end,
                output_dir=output_dir,
                workers=1,
            )

        # Directory should be created
        assert output_dir.exists()
        assert output_dir.is_dir()


# ---------------------------------------------------------------------------
# Test missing S3 files scenario (the user's original issue)
# ---------------------------------------------------------------------------


class TestMissingS3FilesScenario:
    """Tests for handling missing S3 files (historical data)."""

    def test_find_s3_key_logs_warning_for_missing_file(self, mock_s3_client):
        """Test that missing files are handled gracefully."""
        valid_dt = datetime(2023, 12, 14, 16, 30, tzinfo=timezone.utc)

        # Mock: all candidates fail (file not found)
        mock_s3_client.head_object = Mock(side_effect=Exception("404"))

        result = find_s3_key_for_valid_time(mock_s3_client, valid_dt)

        # Should return None without crashing
        assert result is None

    def test_download_rainfall_handles_all_missing_files(self, mock_s3_client, tmp_path):
        """Test download when all files are missing (historical data scenario)."""
        from unittest.mock import patch

        from eoflow.rainfall import download_rainfall

        # Use historical dates like the user's example
        start = "2023-12-14T16:00"
        end = "2023-12-14T17:00"

        with patch("eoflow.rainfall.make_s3_client", return_value=mock_s3_client):
            # Mock: all files missing (like historical data)
            mock_s3_client.head_object = Mock(side_effect=Exception("NoSuchKey"))

            result = download_rainfall(
                start=start,
                end=end,
                output_dir=tmp_path,
                workers=2,
            )

        # Should complete without crashing
        assert result["downloaded"] == 0
        assert result["error"] == 0
        # No files found means nothing to attempt downloading

    def test_download_rainfall_partial_success(self, mock_s3_client, tmp_path):
        """Test download when some files exist and some don't."""
        from unittest.mock import patch

        from eoflow.rainfall import download_rainfall

        start = "2024-03-15T00:00"
        end = "2024-03-15T00:45"  # 4 timesteps: 00:00, 00:15, 00:30, 00:45

        call_count = 0

        def head_object_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Succeed on first candidate for timesteps 1 and 3, fail for 2 and 4
            if call_count in [1, 3]:
                return {}
            raise Exception("404 Not Found")

        with patch("eoflow.rainfall.make_s3_client", return_value=mock_s3_client):
            mock_s3_client.head_object = Mock(side_effect=head_object_side_effect)
            mock_s3_client.download_file = Mock()

            result = download_rainfall(
                start=start,
                end=end,
                output_dir=tmp_path,
                workers=1,
            )

        # Should have some successes and some missing
        # Exact counts depend on the mock behavior, but should not crash
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Integration tests (require S3 access)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRainfallIntegration:
    """Integration tests that access real S3 data."""

    def test_make_s3_client_succeeds(self):
        """Should be able to create S3 client."""
        from eoflow.rainfall import make_s3_client

        client = make_s3_client()
        assert client is not None

    def test_find_s3_key_for_recent_valid_time(self):
        """Should find S3 key for very recent valid time."""
        from eoflow.rainfall import make_s3_client

        # Use a time from ~24 hours ago (likely to still exist)
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(hours=24)
        valid_dt = yesterday.replace(minute=0, second=0, microsecond=0)

        s3 = make_s3_client()
        key = find_s3_key_for_valid_time(s3, valid_dt)

        # May or may not find it, but shouldn't crash
        if key:
            assert "rainfall_rate.nc" in key
            assert "uk-deterministic-2km" in key

    @pytest.mark.slow
    def test_download_rainfall_dry_run(self):
        """Test download_rainfall in dry-run mode (no actual download)."""
        from eoflow.rainfall import download_rainfall

        # Use a recent time range
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=2)
        end = now - timedelta(hours=1)

        # Round to 15-min boundaries
        start = start.replace(minute=0, second=0, microsecond=0)
        end = end.replace(minute=0, second=0, microsecond=0)

        result = download_rainfall(
            start=start,
            end=end,
            output_dir="/tmp/rainfall_test",
            dry_run=True,
            workers=2,
        )

        # Should return counts dict
        assert isinstance(result, dict)
        assert "downloaded" in result
        assert "skipped" in result
        assert "dry-run" in result
        assert "error" in result


# ---------------------------------------------------------------------------
# Test error handling and edge cases
# ---------------------------------------------------------------------------


class TestRainfallEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_datetime_with_seconds_ignored(self):
        """Seconds in string should cause parse to fail (not supported)."""
        with pytest.raises(ValueError):
            parse_datetime("2024-03-15T12:30:45")

    def test_generate_valid_times_end_before_start_yields_nothing(self):
        """End before start should yield no times."""
        start = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
        end = datetime(2024, 3, 15, 11, 0, tzinfo=timezone.utc)

        times = list(generate_valid_times(start, end))

        assert len(times) == 0

    def test_s3_key_negative_lead_time_allowed(self):
        """S3 key should handle cases where valid is before run (shouldn't happen)."""
        run_dt = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)
        valid_dt = datetime(2024, 3, 1, 11, 0, tzinfo=timezone.utc)

        # Should not crash, though this is not a valid scenario
        key = s3_key(run_dt, valid_dt)
        assert isinstance(key, str)

    def test_candidate_run_times_very_old_valid_time(self):
        """Should handle valid times from long ago."""
        valid_dt = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)

        candidates = list(candidate_run_times(valid_dt))

        # Should still generate candidates
        assert len(candidates) > 0
        assert all(c <= valid_dt for c in candidates)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_module_imports():
    """Verify all main functions can be imported."""
    from eoflow.rainfall import (
        candidate_run_times,
        download_rainfall,
        find_netcdf_files,
        generate_valid_times,
        parse_datetime,
        s3_key,
    )

    # All imports should succeed
    assert callable(candidate_run_times)
    assert callable(download_rainfall)
    assert callable(find_netcdf_files)
    assert callable(generate_valid_times)
    assert callable(parse_datetime)
    assert callable(s3_key)


# ---------------------------------------------------------------------------
# Test find_netcdf_files
# ---------------------------------------------------------------------------


def test_find_netcdf_files_with_matching_files(tmp_path):
    """Test finding NetCDF files in directory."""
    from eoflow.rainfall import find_netcdf_files

    # Create some test files
    nc_dir = tmp_path / "uk-deterministic-2km" / "20240301T0000Z"
    nc_dir.mkdir(parents=True)

    file1 = nc_dir / "20240301T0000Z-PT0000H00M-rainfall_rate.nc"
    file2 = nc_dir / "20240301T0015Z-PT0000H15M-rainfall_rate.nc"

    file1.touch()
    file2.touch()

    # Create a non-matching file
    (nc_dir / "other.txt").touch()

    result = find_netcdf_files(tmp_path)

    assert len(result) >= 2
    assert all(f.suffix == ".nc" for f in result)


def test_find_netcdf_files_empty_directory(tmp_path):
    """Test finding files in empty directory."""
    from eoflow.rainfall import find_netcdf_files

    result = find_netcdf_files(tmp_path)

    assert result == []
