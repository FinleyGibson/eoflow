#!/usr/bin/env bash
# Process all locally downloaded NIMROD tar files for every year found in
# data/nimrod_raw/.
#
# For each YYYY sub-directory the script calls nimrod_process_local.py,
# which unpacks the tar files, crops to the shapefile, and writes per-timestep
# NetCDF files under data/nimrod_processed/raw/YYYY/.
#
# Usage:
#   bash scripts/process_all_nimrod.sh
#
# Optional environment variables:
#   SHAPEFILE   - path to shapefile (default: data/shapefiles/devon_county/devon_county.shp)
#   OUTPUT_DIR  - root output directory  (default: data/nimrod_processed/raw)
#   WORKERS     - number of parallel worker threads (default: let the script decide)

set -euo pipefail

SHAPEFILE="${SHAPEFILE:-./data/shapefiles/devon_county/devon_county.shp}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/nimrod_processed/raw}"
WORKERS="${WORKERS:-}"

RAW_DIR="./data/nimrod_raw"

if [ ! -f "$SHAPEFILE" ]; then
    echo "ERROR: Shapefile not found: $SHAPEFILE" >&2
    exit 1
fi

if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: Raw data directory not found: $RAW_DIR" >&2
    exit 1
fi

# Collect year directories, sorted oldest-first
YEAR_DIRS=($(ls -d "${RAW_DIR}"/*/ 2>/dev/null | sort))

if [ ${#YEAR_DIRS[@]} -eq 0 ]; then
    echo "No year directories found in ${RAW_DIR}." >&2
    exit 1
fi

WORKERS_ARG=""
if [ -n "$WORKERS" ]; then
    WORKERS_ARG="--workers ${WORKERS}"
fi

for YEAR_DIR in "${YEAR_DIRS[@]}"; do
    YEAR=$(basename "$YEAR_DIR")
    YEAR_OUTPUT="${OUTPUT_DIR}/${YEAR}"

    echo "=========================================="
    echo "Processing NIMROD data for year: ${YEAR}"
    echo "Input directory:  ${YEAR_DIR}"
    echo "Output directory: ${YEAR_OUTPUT}"
    echo "=========================================="

    uv run scripts/nimrod_process_local.py \
        --input "${YEAR_DIR}" \
        --shapefile "${SHAPEFILE}" \
        --output "${YEAR_OUTPUT}" \
        ${WORKERS_ARG}

    echo "Finished year: ${YEAR}"
    echo ""
done

echo "All years processed successfully."
