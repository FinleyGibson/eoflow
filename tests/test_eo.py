"""
Tests for the Earth Observation API module.

Tests cover:
- API initialization and configuration
"""

from eoflow import eo


class TestOpenEOAPIInit:
    """Tests for API initialization."""

    def test_connection(self):
        api = eo.connect()

        result = api.authenticate_bearer_token()
        a = 1
        print(api)
