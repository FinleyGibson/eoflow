#!/usr/bin/env python
"""
Script to run the Water Quality Samples API server locally.

Usage:
    python run_api.py

The API will be available at http://localhost:8000
API documentation will be available at http://localhost:8000/docs
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.eoflow.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
