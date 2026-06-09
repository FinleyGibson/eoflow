#!/usr/bin/sh

# Load CEDA_TOKEN from .env in the repo root (one level above this script).
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$ENV_FILE"
    set +a
fi

if [ -z "$CEDA_TOKEN" ]; then
    echo "Error: CEDA_TOKEN is not set. Add it to .env or export it in your shell."
    exit 1
fi

RESPONSE=$(wget -O - -q \
    --header "Authorization: Bearer $CEDA_TOKEN" \
    https://dap.ceda.ac.uk/badc/ARCHIVE_INFO/ACCESS_TEST/RESTRICTED/TOKEN_CHECK 2>&1)

if echo "$RESPONSE" | grep -qi "<html"; then
    echo "Authentication failed: received  a login redirect."
    echo "Check your CEDA_TOKEN in .env. or set using:"
    echo "export CEDA_TOKEN=xxx <-token here"
    exit 1
else
    echo "Authentication successful."
    echo "Response: $RESPONSE"
fi
