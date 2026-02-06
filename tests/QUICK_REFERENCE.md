# Test Quick Reference

## Run All Tests (Unit Only - Default)

```bash
pytest
```

By default, integration tests are **excluded** to keep tests fast.

## Run Integration Tests Only

```bash
# Basic
pytest -m integration

# With verbose output and print statements visible
pytest -m integration -v -s

# Specific integration test file
pytest tests/api/test_api_ea_integration.py

# Specific test
pytest -m integration -k "test_get_data_with_area_none" -v -s
```

## Run Tests in Specific Directory

```bash
# All tests in api directory
pytest tests/api

# Just unit tests in api directory
pytest tests/api -m "not integration"

# Just integration tests in api directory
pytest tests/api -m integration
```

## Run Specific Test File

```bash
pytest tests/api/test_api_ea.py
pytest tests/api/test_api_ea_integration.py
```

## Run Specific Test Class or Function

```bash
# Specific class
pytest tests/api/test_api_ea.py::TestEAWaterQualityEndpoint

# Specific test function
pytest tests/api/test_api_ea.py::TestEAWaterQualityEndpoint::test_ea_water_quality_success

# Integration test
pytest tests/api/test_api_ea_integration.py::TestEAAPIIntegration::test_get_data_with_area_none -v -s
```

## Useful Flags

```bash
# Verbose output (show test names)
pytest -v

# Show print statements (don't capture output)
pytest -s

# Stop at first failure
pytest -x

# Show test durations (find slow tests)
pytest --durations=10

# Run tests matching pattern
pytest -k "water_quality"

# Run with coverage
pytest --cov=eoflow.ea
```

## Common Combinations

```bash
# Unit tests only (fast)
pytest -m "not integration"

# Fast tests only (exclude slow and integration)
pytest -m "not slow and not integration"

# Integration tests with output
pytest -m integration -v -s

# Run all tests including integration
pytest -m "integration or not integration"

# Specific file with verbose output and stop on first failure
pytest tests/api/test_api_ea.py -v -x

# API tests only (no integration)
pytest tests/api -m "not integration" -v
```

## Test Markers

- `@pytest.mark.integration` - Tests that call real external APIs
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.skip` - Tests that are always skipped

## Examples

### Daily Development

```bash
# Quick test run (unit tests only)
pytest

# Test specific feature
pytest tests/api/test_api_ea.py -v
```

### Before Committing

```bash
# Run all unit tests with verbose output
pytest -v

# Check specific module
pytest tests/api -v
```

### Integration Testing

```bash
# Run integration tests to verify EA API works
pytest -m integration -v -s

# Run specific integration test
pytest -m integration -k "area_none" -v -s
```

### Debugging

```bash
# Run one test with full output and stop on failure
pytest tests/api/test_api_ea.py::TestEAWaterQualityEndpoint::test_ea_water_quality_success -v -s -x

# See what tests would run without executing them
pytest --collect-only

# See which tests match a pattern
pytest --collect-only -k "polygon"
```

## Pro Tips

1. **Use `-v -s` together** when debugging to see test names and print output
2. **Use `-x`** to stop at first failure when fixing bugs
3. **Use `-k "pattern"`** to run tests matching a name pattern
4. **Skip integration tests** during rapid development: `pytest -m "not integration"`
5. **Run integration tests** before major releases or when testing API changes

## Configuration

Tests are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests that call external APIs",
]
```

## More Information

- Full integration test docs: `tests/INTEGRATION_TESTS.md`
- Pytest docs: https://docs.pytest.org/
