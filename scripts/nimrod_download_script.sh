#!/usr/bin/sh

# Load CEDA_TOKEN from .env in the repo root (one level above this script).
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    # Export only lines of the form KEY=VALUE, ignoring comments and blanks.
    set -a
    # shellcheck source=/dev/null
    . "$ENV_FILE"
    set +a
fi

usage() {
    echo "Usage: $0 <year> <target_directory>"
    echo ""
    echo "  <year>              Four-digit year to download, e.g. 2023"
    echo "  <target_directory>  Directory to save downloaded files into"
    echo ""
    echo "  Requires CEDA_TOKEN to be set - either in .env (repo root) or exported in your shell."
    echo ""
    echo "Example:"
    echo "  $0 2023 /data/nimrod"
    exit 1
}

if [ -z "$1" ] || [ -z "$2" ]; then
    usage
fi

if [ -z "$CEDA_TOKEN" ]; then
    echo "Error: CEDA_TOKEN environment variable is not set."
    exit 1
fi

YEAR="$1"
TARGET_DIR="$2"
URL="https://dap.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km/$YEAR/"

echo "Downloading Nimrod data for: $YEAR"
echo "Saving to:                   $TARGET_DIR"
echo ""

mkdir -p "$TARGET_DIR"

wget \
    -e robots=off \
    --mirror \
    --no-parent \
    -np \
    -r \
    -P "$TARGET_DIR" \
    --header "Authorization: Bearer $CEDA_TOKEN" \
    "$URL"
