# EOFlow API Documentation

## Overview

The EOFlow API provides RESTful endpoints for retrieving water quality data from multiple sources:

1. **Water Quality Samples** - Internal water quality sample data for geographic points
2. **EA Water Quality Data** - Environment Agency monitoring data filtered by polygon boundaries

## Getting Started

### Starting the API Server

```bash
# Development mode with auto-reload
uvicorn eoflow.api:app --reload

# Production mode
uvicorn eoflow.api:app --host 0.0.0.0 --port 8000
```

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Root Endpoint

```
GET /
```

Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Water Quality API",
  "version": "1.0.0",
  "endpoints": {
    "/samples": "POST - Get water quality samples for a point",
    "/ea/water-quality": "POST - Get EA water quality data within a polygon",
    "/ea/water-quality/multiple": "POST - Get multiple EA determinands within a polygon",
    "/docs": "GET - Interactive API documentation"
  }
}
```

### Health Check

```
GET /health
```

Returns health status of the API.

**Response:**
```json
{
  "status": "healthy"
}
```

---

## Water Quality Samples Endpoints

### Get Samples for Point

```
POST /samples
```

Retrieves water quality samples for a given geographic point from the catchment area containing that point.

**Request Body:**
```json
{
  "longitude": -1.5,
  "latitude": 52.5
}
```

**Response:**
```json
[
  {
    "id": "sample_001",
    "long": -1.5,
    "lat": 52.5,
    "water_body_type": "stream",
    "date": "2024-01-15",
    "temperature": 12.5,
    "ph": 7.2,
    "turbidity": 5.3,
    "phosphate": 0.15,
    "nitrate": "2.5",
    "ammonia": 0.1,
    "pollution_evidence": ["plastic"],
    "wildlife": ["ducks", "fish"],
    "bank_vegetation": ["grass", "trees"],
    "land_use": ["agriculture"]
  }
]
```

---

## EA Water Quality Endpoints

### Get EA Water Quality Data (Single Determinand)

```
POST /ea/water-quality
```

Fetches water quality data from the Environment Agency API for a specific determinand and filters results to observations within the provided polygon.

**Request Body:**
```json
{
  "polygon": {
    "coordinates": [
      [-4.5, 50.3],
      [-4.5, 51.2],
      [-3.0, 51.2],
      [-3.0, 50.3],
      [-4.5, 50.3]
    ]
  },
  "determinand": "0076",
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "area": "environment_agency,SWX",
  "verbose": false
}
```

**Parameters:**
- `polygon.coordinates`: List of [longitude, latitude] pairs defining the polygon (minimum 3 points)
- `determinand`: Determinand code (see Common Determinands below)
- `start_date`: Start date in YYYY-MM-DD format
- `end_date`: End date in YYYY-MM-DD format
- `area`: Precanned area code (see Common Area Codes below)
- `verbose`: Optional, whether to show verbose logging (default: false)

**Response:**
```json
{
  "total_records": 150,
  "filtered_records": 42,
  "data": [
    {
      "id": "...",
      "result": "15.5",
      "phenomenonTime": "2024-01-01T10:00:00",
      "Date": "2024-01-01",
      "sample.samplingPoint.notation": "SP001",
      "sample.samplingPoint.latitude": "50.5",
      "sample.samplingPoint.longitude": "-4.0",
      "determinand.notation": "0076",
      "determinand.prefLabel": "Temperature of Water",
      "unit": "°C"
    }
  ]
}
```

### Get EA Water Quality Data (Multiple Determinands)

```
POST /ea/water-quality/multiple
```

Fetches water quality data for multiple determinands, joins them together, and filters results to observations within the provided polygon.

**Request Body:**
```json
{
  "polygon": {
    "coordinates": [
      [-4.5, 50.3],
      [-4.5, 51.2],
      [-3.0, 51.2],
      [-3.0, 50.3],
      [-4.5, 50.3]
    ]
  },
  "determinands": {
    "0076": "Temperature",
    "0077": "Conductivity",
    "0180": "Orthophosphate"
  },
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "area": "environment_agency,SWX",
  "verbose": false
}
```

**Parameters:**
- `polygon.coordinates`: List of [longitude, latitude] pairs defining the polygon (minimum 3 points)
- `determinands`: Dictionary mapping determinand codes to column names
- `start_date`: Start date in YYYY-MM-DD format
- `end_date`: End date in YYYY-MM-DD format
- `area`: Precanned area code (see Common Area Codes below)
- `verbose`: Optional, whether to show verbose logging (default: false)

**Response:**
```json
{
  "total_records": 150,
  "filtered_records": 42,
  "determinands": ["0076", "0077", "0180"],
  "data": [
    {
      "phenomenonTime": "2024-01-01T10:00:00",
      "Date": "2024-01-01",
      "Temperature": "15.5",
      "Conductivity": "100",
      "Orthophosphate": "0.15",
      "sample.samplingPoint.notation": "SP001",
      "sample.samplingPoint.latitude": "50.5",
      "sample.samplingPoint.longitude": "-4.0"
    }
  ]
}
```

---

## Reference Data

### Common Determinands

| Code | Parameter                     | Unit |
|------|-------------------------------|------|
| 0076 | Temperature of Water          | °C   |
| 0077 | Conductivity at 25°C          | µS/cm|
| 0180 | Orthophosphate, reactive as P | mg/l |
| 0115 | Dissolved oxygen saturation   | %    |
| 0556 | Nitrate as N                  | mg/l |
| 6396 | Turbidity (NTU)               | NTU  |
| 0117 | pH                            | pH   |
| 0191 | Total oxidised nitrogen as N  | mg/l |

### Common Area Codes

| Region     | Code                      |
|------------|---------------------------|
| Southwest  | environment_agency,SWX    |
| DCS        | environment_agency,DCS    |
| Thames     | environment_agency,TH     |
| Anglian    | environment_agency,AN     |
| Midlands   | environment_agency,MD     |
| North East | environment_agency,NE     |
| North West | environment_agency,NW     |
| Southern   | environment_agency,SN     |
| Yorkshire  | environment_agency,YK     |

---

## Error Responses

### 400 Bad Request

Invalid request parameters.

```json
{
  "detail": "Invalid request parameters: 'area' parameter must be provided"
}
```

### 422 Unprocessable Entity

Validation error (e.g., missing required fields, invalid data types).

```json
{
  "detail": [
    {
      "loc": ["body", "polygon", "coordinates"],
      "msg": "ensure this value has at least 3 items",
      "type": "value_error.list.min_items"
    }
  ]
}
```

### 500 Internal Server Error

Server error or missing dependencies.

```json
{
  "detail": "An error occurred while processing your request: ..."
}
```

### 501 Not Implemented

Feature not yet implemented.

```json
{
  "detail": "The water quality sampling functionality is not yet fully implemented."
}
```

---

## Example Usage

### Python (requests)

```python
import requests

# Single determinand
response = requests.post(
    "http://localhost:8000/ea/water-quality",
    json={
        "polygon": {
            "coordinates": [
                [-4.5, 50.3],
                [-4.5, 51.2],
                [-3.0, 51.2],
                [-3.0, 50.3],
                [-4.5, 50.3]
            ]
        },
        "determinand": "0076",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "area": "environment_agency,SWX"
    }
)

data = response.json()
print(f"Found {data['filtered_records']} records")
```

### cURL

```bash
curl -X POST "http://localhost:8000/ea/water-quality" \
  -H "Content-Type: application/json" \
  -d '{
    "polygon": {
      "coordinates": [
        [-4.5, 50.3],
        [-4.5, 51.2],
        [-3.0, 51.2],
        [-3.0, 50.3],
        [-4.5, 50.3]
      ]
    },
    "determinand": "0076",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "area": "environment_agency,SWX"
  }'
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/ea/water-quality', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    polygon: {
      coordinates: [
        [-4.5, 50.3],
        [-4.5, 51.2],
        [-3.0, 51.2],
        [-3.0, 50.3],
        [-4.5, 50.3]
      ]
    },
    determinand: '0076',
    start_date: '2024-01-01',
    end_date: '2024-01-31',
    area: 'environment_agency,SWX'
  })
});

const data = await response.json();
console.log(`Found ${data.filtered_records} records`);
```

---

## Notes

- **Rate Limiting**: The EA API client includes built-in rate limiting with configurable delays between requests
- **Automatic Pagination**: Large date ranges are automatically split into monthly chunks
- **Data Filtering**: Records with missing results are automatically dropped
- **Polygon Format**: Polygons must be closed (first and last coordinate pairs should be the same)
- **Date Format**: All dates must be in YYYY-MM-DD format
- **Dependencies**: The `shapely` package is required for polygon filtering

---

## See Also

- [EA Water Quality API Documentation](https://environment.data.gov.uk/water-quality-beta/api-docs)
- [Example Scripts](../examples/)
- [Configuration Guide](../README.md)
