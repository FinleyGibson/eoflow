# EOFlow Examples

This directory contains example scripts demonstrating how to use the various components of the EOFlow package.

## Available Examples

### EA Water Quality API - Quick Reference (`ea_quick_reference.py`)

A concise, copy-paste ready reference guide with minimal examples for common tasks.

**Quick examples for:**

- Simple queries using the convenience function
- Using the API class with custom settings
- Fetching multiple determinands
- Polygon filtering
- Working with results
- Error handling

**Best for:** Quick lookups and copy-paste code snippets.

### EA Water Quality API - Detailed Examples (`ea_water_quality_example.py`)

Comprehensive examples demonstrating how to fetch and analyze water quality data from the Environment Agency's Water Quality API.

**Examples included:**

1. **Simple Query** - Using the convenience function to fetch single determinand data
2. **API Class Usage** - Using the API class directly with custom settings
3. **Multiple Determinands** - Fetching and combining multiple water quality parameters
4. **Polygon Filtering** - Filtering observations by geographic boundaries
5. **Data Analysis** - Basic statistical analysis of water quality data

**Requirements:**

- Network access to the EA API
- Optional: `shapely` package for polygon filtering (`pip install shapely`)

**Usage:**

```bash
# View quick reference (no execution, just code snippets)
cat examples/ea_quick_reference.py

# Run detailed examples
python examples/ea_water_quality_example.py

# Or import and run specific examples
python -c "from examples.ea_water_quality_example import example_1_simple_query; example_1_simple_query()"
```

## Common Determinands

| Code | Parameter                     |
| ---- | ----------------------------- |
| 0076 | Temperature of Water          |
| 0077 | Conductivity at 25°C          |
| 0180 | Orthophosphate, reactive as P |
| 6396 | Turbidity (NTU)               |

## Common Area Codes

| Code                        | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| `environment_agency,DCS`    | Environment Agency DCS area                          |
| `environment_agency,SWX`    | Southwest region (Devon, Cornwall, Somerset, Dorset) |
| `local_authority,E06000002` | Example local authority area                         |

## API Documentation

- [EA Water Quality API Documentation](https://environment.data.gov.uk/water-quality-beta/api-docs)
- [API Usage Guide (Gist)](https://gist.github.com/canwaf/2afa25fc6160efb25ac72b7acd60278d)

## Notes

- The examples use real API endpoints and require network connectivity
- API requests are rate-limited with delays between calls (configurable)
- Some queries may return no data depending on availability for the specified date range and area
- Large date ranges will automatically be split into monthly chunks to handle API limitations
