#! /usr/bin/sh

usage() {
    echo "Usage: $0 <date> <target_directory>"
    echo ""
    echo "  <date>              Date path to download, e.g. 2023 or 2023/01"
    echo "  <target_directory>  Directory to save downloaded files into"
    echo ""
    echo "  Requires the CEDA_TOKEN environment variable to be set."
    echo ""
    echo "Example:"
    echo "  CEDA_TOKEN=mytoken $0 2023/01 /data/nimrod"
    exit 1
}

if [ -z "$1" ] || [ -z "$2" ]; then
    usage
fi

if [ -z "$CEDA_TOKEN" ]; then
    echo "Error: CEDA_TOKEN environment variable is not set."
    exit 1
fi

DATE="$1"
TARGET_DIR="$2"
URL="https://dap.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km/$DATE/"

echo "Downloading Nimrod data for: $DATE"
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
