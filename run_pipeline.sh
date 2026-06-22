#!/usr/bin/env bash
# ==========================================================================
# run_pipeline.sh
# --------------------------------------------------------------------------
# End-to-end eoflow dataset pipeline.  Runs each step in sequence; the
# next step only starts after the previous one finishes successfully.
#
# Usage:
#   ./run_pipeline.sh                     # use all defaults
#   ./run_pipeline.sh --skip-eo           # skip the slow openEO fetch
#   ./run_pipeline.sh --determinand 6396  # fetch turbidity instead
#
# Configurable via environment variables (or edit the defaults below).
# ==========================================================================
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Colour

step_start() { echo -e "\n${CYAN}══════════════════════════════════════════════════════════════${NC}"; echo -e "${CYAN}  STEP $1 — $2${NC}"; echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}\n"; }
step_ok()    { echo -e "\n${GREEN}  ✓ Step $1 completed successfully.${NC}\n"; }
step_skip()  { echo -e "\n${YELLOW}  ⊘ Step $1 skipped ($2).${NC}\n"; }

# ── Configurable defaults ────────────────────────────────────────────────
SHAPEFILE="${SHAPEFILE:-data/shapefiles/devon_county}"
DETERMINAND="${DETERMINAND:-0076}"
START_DATE="${START_DATE:-2023-01-01}"
END_DATE="${END_DATE:-2023-12-31}"
WQ_CSV="${WQ_CSV:-data/wq_samples/devon_turbidity_2023.csv}"
DEM="${DEM:-data/dems/devon_dem_cop30.tif}"
GPKG="${GPKG:-outputs/devon_catchments.gpkg}"
SAMPLES_DIR="${SAMPLES_DIR:-data/sample_instances}"
FEATURES_CSV="${FEATURES_CSV:-outputs/catchment_features.csv}"
VIS_HTML="${VIS_HTML:-outputs/devon_turbidity_map.html}"
LAT_COL="${LAT_COL:-latitude}"
LON_COL="${LON_COL:-longitude}"
FLOW_ACC="${FLOW_ACC:-1000}"

# Optional flags
SKIP_FETCH="${SKIP_FETCH:-false}"
SKIP_DEM="${SKIP_DEM:-false}"
SKIP_EO="${SKIP_EO:-false}"
SKIP_VIS="${SKIP_VIS:-false}"
SAVE_PARQUET="${SAVE_PARQUET:-true}"

# Parse CLI overrides
for arg in "$@"; do
    case "$arg" in
        --skip-fetch)   SKIP_FETCH=true ;;
        --skip-dem)     SKIP_DEM=true ;;
        --skip-eo)      SKIP_EO=true ;;
        --skip-vis)     SKIP_VIS=true ;;
        --no-parquet)   SAVE_PARQUET=false ;;
        --determinand=*) DETERMINAND="${arg#*=}" ;;
        --determinand)   shift_next=determinand ;;
        *)
            if [ "${shift_next:-}" = "determinand" ]; then
                DETERMINAND="$arg"
                shift_next=""
            fi
            ;;
    esac
done

echo -e "${CYAN}eoflow dataset pipeline${NC}"
echo "  shapefile    : $SHAPEFILE"
echo "  determinand  : $DETERMINAND"
echo "  date range   : $START_DATE → $END_DATE"
echo "  WQ CSV       : $WQ_CSV"
echo "  DEM          : $DEM"
echo "  GeoPackage   : $GPKG"
echo "  samples dir  : $SAMPLES_DIR"
echo "  features CSV : $FEATURES_CSV"
echo "  skip fetch   : $SKIP_FETCH"
echo "  skip DEM     : $SKIP_DEM"
echo "  skip EO      : $SKIP_EO"

# ==========================================================================
# Step 1 — Fetch water-quality data from the EA API
# ==========================================================================
if [ "$SKIP_FETCH" = "true" ]; then
    step_skip 1 "SKIP_FETCH=true"
elif [ -f "$WQ_CSV" ]; then
    step_skip 1 "$WQ_CSV already exists"
else
    step_start 1 "Fetch water-quality data from the EA API"
    python -m scripts.ge_ea_water_quality_by_shapefile \
        --shapefile   "$SHAPEFILE" \
        --determinand "$DETERMINAND" \
        --start-date  "$START_DATE" \
        --end-date    "$END_DATE" \
        --out         "$WQ_CSV"
    step_ok 1
fi

# ==========================================================================
# Step 2 — Download DEM (optional)
# ==========================================================================
if [ "$SKIP_DEM" = "true" ]; then
    step_skip 2 "SKIP_DEM=true"
elif [ -f "$DEM" ]; then
    step_skip 2 "$DEM already exists"
else
    step_start 2 "Download DEM"
    python -m scripts.get_devon_dem --output "$DEM"
    step_ok 2
fi

# ==========================================================================
# Step 3 — Build the catchment dataset
# ==========================================================================
if [ -f "$GPKG" ]; then
    step_skip 3 "$GPKG already exists — will resume/reuse"
fi
step_start 3 "Delineate catchments for dataset"
python -m scripts.delineate_catchments \
    --csv  "$WQ_CSV" \
    --dem  "$DEM" \
    --out  "$GPKG" \
    --lat-col "$LAT_COL" \
    --lon-col "$LON_COL" \
    --flow-acc-threshold "$FLOW_ACC"
step_ok 3

# ==========================================================================
# Step 4 — Prepare sample instances (compute layers + fetch EO)
# ==========================================================================
step_start 4 "Prepare sample instances"
PREPARE_ARGS=(
    --gpkg    "$GPKG"
    --dem     "$DEM"
    --out-dir "$SAMPLES_DIR"
    --rainfall-start "$START_DATE"
    --rainfall-end   "$END_DATE"
    --eo-start       "$START_DATE"
    --eo-end         "$END_DATE"
)
if [ "$SKIP_EO" = "true" ]; then
    PREPARE_ARGS+=(--skip-eo)
fi
python -m scripts.prepare_samples "${PREPARE_ARGS[@]}"
step_ok 4

# ==========================================================================
# Step 5 — Extract features
# ==========================================================================
step_start 5 "Extract features"
EXTRACT_ARGS=(
    --samples-dir "$SAMPLES_DIR"
    --output      "$FEATURES_CSV"
)
if [ "$SAVE_PARQUET" = "true" ]; then
    EXTRACT_ARGS+=(--parquet)
fi
python -m scripts.extract_features "${EXTRACT_ARGS[@]}"
step_ok 5

# ==========================================================================
# Step 6 — Visualise sample points
# ==========================================================================
if [ "$SKIP_VIS" = "true" ]; then
    step_skip 6 "SKIP_VIS=true"
else
    step_start 6 "Visualise sample points"
    python -m scripts.visualise_ea_samples \
        --input  "$WQ_CSV" \
        --output "$VIS_HTML"
    step_ok 6
fi

# ==========================================================================
# Done
# ==========================================================================
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Pipeline complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Outputs:"
echo "    Water quality CSV : $WQ_CSV"
echo "    Catchments GPKG   : $GPKG"
echo "    Sample instances  : $SAMPLES_DIR"
echo "    Features CSV      : $FEATURES_CSV"
if [ "$SAVE_PARQUET" = "true" ]; then
    echo "    Features Parquet  : ${FEATURES_CSV%.csv}.parquet"
fi
if [ "$SKIP_VIS" != "true" ]; then
    echo "    Visualisation     : $VIS_HTML"
fi
echo ""
