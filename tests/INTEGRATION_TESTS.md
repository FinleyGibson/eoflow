# Integration Tests

## Overview

Integration tests call the real Environment Agency Water Quality API and are marked with `@pytest.mark.integration`.

These tests are separated from unit tests because they:
- Make real HTTP requests to external APIs
- Are slower than unit tests
- May hit API rate limits
- Require network connectivity
- Can fail due to external service issues

## Running Integration Tests

### Basic Commands

```bash
# Run ONLY integration tests
pytest -m integration

# Run with verbose output and show print statements
pytest -m integration -v -s

# Run specific integration test file
pytest tests/api/test_api_ea_integration.py

# Run specific integration test
pytest -m integration -k "test_get_data_real_api"

# Run all tests EXCEPT integration (default behavior)
pytest -m "not integration"

# Run without integration and without slow tests
pytest -m "not integration and not slow"
```

### Advanced Usage

```bash
# Run integration tests with coverage
pytest -m integration --cov=eoflow.ea

# Run and stop at first failure
pytest -m integration -x

# Run with timing information
pytest -m integration --durations=10

# Run specific test class
pytest -m integration tests/api/test_api_ea_integration.py::TestEAAPIIntegration

# Run specific test method
pytest -m integration tests/api/test_api_ea_integration.py::TestEAAPIIntegration::test_get_data_with_area_none -v -s
```

## Why Separate Integration Tests?

### 1. **Speed**
- Unit tests: milliseconds
- Integration tests: seconds to minutes (real API calls)

### 2. **Reliability**
- Unit tests: 100% reliable (mocked)
- Integration tests: dependent on external service availability

### 3. **Rate Limits**
- Avoid hitting EA API rate limits during normal development
- Run integration tests only when necessary

### 4. **Network Dependency**
- Unit tests can run offline
- Integration tests require internet connection

### 5. **CI/CD Flexibility**
- Can skip integration tests in CI pipelines
- Run integration tests on schedule or manually

## Test Coverage

### TestEAAPIIntegration

Integration tests for core EA API functionality:

| Test | Description | API Calls |
|------|-------------|-----------|
| `test_get_data_real_api_short_range` | Fetches real data (2 days) | 1 |
| `test_get_data_with_area_none` | Tests area=None parameter works | 1 |
| `test_filter_by_polygon_real_data` | Tests polygon filtering with real data | 1 |
| `test_column_structure_real_api` | Verifies API response structure | 1 |
| `test_coordinate_conversion_with_real_data` | Tests easting/northing conversion | 1 |
| `test_southwest_region_filtering` | Tests regional filtering | 1 |
| `test_get_data_full_year` | *(SKIPPED)* Full year query (slow) | 12 |

**Total API calls per test run: ~6**

### TestEAAPIErrorHandling

Error handling tests with real API:

| Test | Description | API Calls |
|------|-------------|-----------|
| `test_invalid_determinand` | Tests handling of invalid codes | 1 |
| `test_future_dates` | Tests querying future dates | 1 |

**Total API calls per test run: ~2**

## What Gets Tested

### ✓ Verified Functionality

1. **API Connectivity**
   - Real HTTP requests to EA API
   - Response parsing and DataFrame creation

2. **Data Retrieval**
   - `get_data()` with `area=None` parameter
   - Short date ranges (1-2 days)
   - Various determinand codes

3. **Polygon Filtering**
   - Auto-detection of easting/northing columns
   - Coordinate conversion (EPSG:27700 → EPSG:4326)
   - Polygon geometry filtering with Shapely

4. **Column Structure**
   - Presence of expected columns
   - Data types and format

5. **Regional Filtering**
   - Filtering by `samplingPoint.region`
   - Southwest region identification

6. **Error Handling**
   - Invalid determinand codes
   - Future dates (no data)
   - Empty responses

## Test Markers

### @pytest.mark.integration

All integration tests are marked with `@pytest.mark.integration`.

```python
@pytest.mark.integration
class TestEAAPIIntegration:
    """Integration tests that call the real EA API."""
    
    def test_something(self, api):
        # This test will only run with: pytest -m integration
        pass
```

### @pytest.mark.slow

Many integration tests are also marked as slow:

```python
@pytest.mark.integration
@pytest.mark.slow
class TestEAAPIIntegration:
    """These tests are both integration and slow."""
    pass
```

### @pytest.mark.skip

Some tests are skipped by default because they make many API calls:

```python
@pytest.mark.skip(reason="This test makes many API calls and is very slow")
def test_get_data_full_year(self, api):
    # This test is always skipped unless you remove the decorator
    pass
```

## Best Practices

### 1. Use Short Date Ranges

```python
# Good - quick test
df = api.get_data(
    determinand="0076",
    start_date="2024-01-01",
    end_date="2024-01-02",  # 2 days
    area=None,
)

# Bad - very slow (12 API calls)
df = api.get_data(
    determinand="0076",
    start_date="2023-01-01",
    end_date="2023-12-31",  # 1 year = 12 months = 12 calls
    area=None,
)
```

### 2. Handle Empty Responses

```python
df = api.get_data(...)

if df.empty:
    pytest.skip("No data returned from API for this date range")

# Continue with assertions
assert "result" in df.columns
```

### 3. Print Informative Messages

```python
if not df.empty:
    print(f"\n✓ Retrieved {len(df)} records from EA API")
else:
    print("\n⚠ No data returned (may be expected)")
```

Run with `-s` flag to see these messages:
```bash
pytest -m integration -s
```

### 4. Use Fixtures for Setup

```python
@pytest.fixture
def api(self):
    """Create API instance with longer timeout for integration tests."""
    return EAWaterQualityAPI(delay=1.0, timeout=60)

def test_something(self, api):
    # Use the fixture
    df = api.get_data(...)
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: pytest -m "not integration"
  
  integration-tests:
    runs-on: ubuntu-latest
    # Only run on schedule or manual trigger
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    steps:
      - uses: actions/checkout@v2
      - name: Run integration tests
        run: pytest -m integration
```

### Makefile Example

```makefile
.PHONY: test test-unit test-integration

test:
	pytest

test-unit:
	pytest -m "not integration"

test-integration:
	pytest -m integration -v -s

test-all:
	pytest -m "integration or not integration"
```

## Troubleshooting

### No Data Returned

**Problem:** Tests skip because `df.empty` is True.

**Solutions:**
- Try different date ranges
- Check EA API status: https://environment.data.gov.uk/water-quality-beta/api-docs
- Verify internet connectivity
- Try a different determinand code

### API Rate Limiting

**Problem:** Tests fail with 429 Too Many Requests.

**Solutions:**
- Increase delay between requests: `EAWaterQualityAPI(delay=2.0)`
- Run fewer tests at once
- Wait before running again

### Slow Tests

**Problem:** Integration tests take too long.

**Solutions:**
- Use shorter date ranges
- Skip specific slow tests: `pytest -m integration -k "not full_year"`
- Run only fast integration tests

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'eoflow'`

**Solution:**
```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/eoflow/src:$PYTHONPATH
```

## Adding New Integration Tests

### Template

```python
@pytest.mark.integration
@pytest.mark.slow  # Optional
class TestMyNewIntegration:
    """Description of test suite."""
    
    @pytest.fixture
    def api(self):
        """Create API instance."""
        return EAWaterQualityAPI(delay=1.0, timeout=60)
    
    def test_my_new_feature(self, api):
        """Test description."""
        # Fetch data
        df = api.get_data(
            determinand="0076",
            start_date="2024-01-01",
            end_date="2024-01-02",
            area=None,
            verbose=False,
        )
        
        # Handle empty response
        if df.empty:
            pytest.skip("No data returned from API")
        
        # Your test logic
        assert something
        
        # Informative output
        print(f"\n✓ Test passed: {details}")
```

### Guidelines

1. **Keep date ranges short** (1-7 days max)
2. **Handle empty responses gracefully** with `pytest.skip()`
3. **Add informative print statements** for `-s` flag
4. **Use appropriate markers** (`@pytest.mark.integration`, `@pytest.mark.slow`)
5. **Skip expensive tests** by default with `@pytest.mark.skip()`
6. **Document API call count** in docstring or comments

## Related Files

- `tests/api/test_api_ea_integration.py` - Integration test implementation
- `tests/api/test_api_ea.py` - Unit tests (mocked)
- `pyproject.toml` - Pytest configuration with markers
- `src/eoflow/ea.py` - EA API client implementation

## Further Reading

- [Pytest Documentation - Markers](https://docs.pytest.org/en/stable/how-to/mark.html)
- [EA Water Quality API Docs](https://environment.data.gov.uk/water-quality-beta/api-docs)
- [Integration Testing Best Practices](https://martinfowler.com/articles/practical-test-pyramid.html)

---

**Last Updated:** February 2026
