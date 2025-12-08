# Water Quality API - Quick Start

## Start the Server

```bash
python run_api.py
```

Server runs at: `http://localhost:8000`

## Make a Request

### Using curl
```bash
curl -X POST "http://localhost:8000/samples" \
  -H "Content-Type: application/json" \
  -d '{"longitude": -1.5, "latitude": 52.5}'
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/samples",
    json={"longitude": -1.5, "latitude": 52.5}
)

samples = response.json()
```

## Interactive Docs

Visit `http://localhost:8000/docs` to test the API in your browser.

## Install Dependencies

```bash
pip install -e .
```
