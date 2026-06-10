#!/usr/bin/env bash
# Download NIMROD data for all years from 2025 down to 2016.
# Usage: bash scripts/download_all_nimrod.sh

set -euo pipefail

YEARS=(2025 2024 2023 2022 2021 2020 2019 2018 2017 2016)

for YEAR in "${YEARS[@]}"; do
    OUTPUT_DIR="./data/nimrod_${YEAR}"
    echo "=========================================="
    echo "Downloading NIMROD data for year: ${YEAR}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "=========================================="
    uv run scripts/nimrod_download_script.sh "${YEAR}" "${OUTPUT_DIR}"
    echo "Finished year: ${YEAR}"
    echo ""
done

echo "All years downloaded successfully."
