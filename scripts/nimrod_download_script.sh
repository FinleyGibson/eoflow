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

# Fail fast rather than discovering an expired token after a long mirror run:
# CEDA tokens are short-lived, so re-check before every download, not just once
# per session. A rejected/expired token gets served back as an HTML login page
# (200 OK, not 401), which wget happily saves as if it were the real file.
AUTH_CHECK="$(dirname "$0")/nimrod_authentication_check.sh"
if [ -x "$AUTH_CHECK" ] && ! "$AUTH_CHECK" >/dev/null 2>&1; then
    echo "Error: CEDA_TOKEN failed authentication check. Refresh it at" >&2
    echo "  https://accounts.ceda.ac.uk/realms/ceda/account/#/ and update .env" >&2
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

# The token can also expire mid-mirror (this run may take 10+ minutes for a
# full year). A lapsed token makes CEDA redirect to an HTML sign-in page,
# which wget follows and silently saves under the original .tar filename -
# same failure mode as an expired token at start, just discovered later.
# Validate every downloaded tar so this is caught here instead of surfacing
# as a confusing failure several pipeline steps downstream.
echo ""
echo "Verifying downloaded tar files..."
bad_files=""
for f in $(find "$TARGET_DIR" -name "*.tar"); do
    if ! tar -tf "$f" >/dev/null 2>&1; then
        bad_files="$bad_files$f
"
    fi
done

if [ -n "$bad_files" ]; then
    bad_count=$(printf '%s' "$bad_files" | grep -c .)
    echo "Error: $bad_count downloaded file(s) are not valid tar archives" >&2
    echo "(likely an expired CEDA_TOKEN mid-download - CEDA serves an HTML" >&2
    echo "login page in place of the file, which wget saves under the" >&2
    echo "original name). Removing them; refresh CEDA_TOKEN and rerun:" >&2
    printf '%s' "$bad_files" | while IFS= read -r f; do
        [ -n "$f" ] && echo "  $f" >&2 && rm -f "$f"
    done
    exit 1
fi

echo "All downloaded tar files verified OK."
