#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: concat_script.sh [-f] <source_dir> <target_dir>

Merges the per-timestep NetCDF files in each day directory
(YYYYMMDD, found at any depth under source_dir) into a single
per-day NetCDF file under target_dir, mirroring source_dir's
directory structure (e.g. year subdirs) and creating it as needed.

By default, days that already have an output file are skipped.

These files store time as a scalar coordinate rather than a
record dimension, so a plain `ncrcat`/`cdo mergetime` silently
keeps only one timestep. This script instead stacks files with
`ncecat` and then repairs the time/forecast_reference_time/
forecast_period coordinates using the timestamps encoded in each
filename (NCO otherwise "fixes" those to the first file's value).

Options:
  -f          Force reprocessing, overwriting existing output files.
  -h, --help  Show this help and exit.

Examples:
  # Merge a single year
  concat_script.sh raw/2020 concatenated/2020

  # Merge everything (all years), creating year subdirs under target
  concat_script.sh raw concatenated

  # Re-merge a single year, overwriting existing output
  concat_script.sh -f raw/2020 concatenated/2020
EOF
}

force=0
args=()
for arg in "$@"; do
    case "$arg" in
        -f) force=1 ;;
        -h|--help) usage; exit 0 ;;
        *) args+=("$arg") ;;
    esac
done

if (( ${#args[@]} != 2 )); then
    usage >&2
    exit 1
fi

SOURCE_DIR="${args[0]%/}"
TARGET_DIR="${args[1]%/}"

while IFS= read -r -d '' day_dir; do
    rel="${day_dir#"$SOURCE_DIR"/}"
    outfile="${TARGET_DIR}/${rel}.nc"

    echo "-> $rel"

    if (( ! force )) && [[ -f "$outfile" ]]; then
        echo "   $outfile already exists, skipping (use -f to force)"
        continue
    fi

    shopt -s nullglob
    files=("$day_dir"/*.nc)
    shopt -u nullglob

    if (( ${#files[@]} == 0 )); then
        echo "   No nc files found, skipping"
        continue
    fi

    # Filenames are YYYYMMDD_HHMMSS.nc; derive the true per-file
    # timestamp from the name rather than trusting NCO to preserve it.
    datestrs=""
    for f in "${files[@]}"; do
        b=$(basename "$f" .nc)
        datestrs+="${b:0:4}-${b:4:2}-${b:6:2} ${b:9:2}:${b:11:2}:${b:13:2}"$'\n'
    done
    epochs=($(printf '%s' "$datestrs" | date -u -f - +%s))

    if (( ${#epochs[@]} != ${#files[@]} )); then
        echo "   Could not parse timestamps for all files, skipping" >&2
        continue
    fi

    mkdir -p "$(dirname "$outfile")"

    tmpfile=$(mktemp "${outfile}.tmp.XXXXXX")
    trap 'rm -f "$tmpfile"' EXIT

    ncecat -O -u time "${files[@]}" "$tmpfile"

    joined=$(IFS=,; echo "${epochs[*]}")
    ncap2 -O -s "
        time[\$time]={${joined}};
        time@units=\"seconds since 1970-01-01 00:00:00\";
        time@standard_name=\"time\";
        time@calendar=\"standard\";
        forecast_reference_time[\$time]=time;
        forecast_reference_time@units=\"seconds since 1970-01-01 00:00:00\";
        forecast_reference_time@standard_name=\"forecast_reference_time\";
        forecast_reference_time@calendar=\"standard\";
        forecast_period[\$time]=0;
        forecast_period@units=\"second\";
        forecast_period@standard_name=\"forecast_period\";
    " "$tmpfile" "$outfile"

    rm -f "$tmpfile"
    trap - EXIT

    echo "   Wrote $outfile"
done < <(find "$SOURCE_DIR" -mindepth 1 -type d -regextype posix-extended -regex '.*/[0-9]{8}' -print0 | sort -z)
