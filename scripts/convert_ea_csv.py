"""
Convert raw EA water-quality CSV into a cleaner format.

Transformations
---------------
1. Rename the ``result`` column to the determinand name found in
   ``determinand.prefLabel`` (e.g. "Turbidity").  When the file contains
   multiple determinands the data is pivoted so that each determinand
   becomes its own column.
2. Add ``latitude`` and ``longitude`` columns (WGS 84) derived from the
   existing ``samplingPoint.easting`` / ``samplingPoint.northing`` values
   (British National Grid, EPSG:27700).
3. Retain the original ``samplingPoint.easting`` and
   ``samplingPoint.northing`` columns.

Usage
-----
    python -m scripts.convert_ea_csv \
        --input ea_script_out.csv \
        --output ea_water_quality_clean.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eoflow.log_utils import get_logger
from eoflow.utils import easting_northing_to_latlon

logger = get_logger(__file__)

# ---- Column groups --------------------------------------------------------

# Columns that identify a unique observation *location + time* (i.e. everything
# except the determinand / result / unit triad).
_ID_COLS = [
    "id",
    "samplingPoint.notation",
    "samplingPoint.prefLabel",
    "samplingPoint.easting",
    "samplingPoint.northing",
    "samplingPoint.region",
    "samplingPoint.area",
    "samplingPoint.subArea",
    "samplingPoint.samplingPointStatus",
    "samplingPoint.samplingPointType",
    "phenomenonTime",
    "samplingPurpose",
    "sampleMaterialType",
]


def _build_column_name(determinand: str, unit: str | None = None) -> str:
    """Return a human-readable column name, optionally including the unit."""
    if unit and str(unit).strip():
        return f"{determinand} ({unit})"
    return determinand


def convert(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Read *input_path*, transform, and write to *output_path*.

    Returns the transformed :class:`~pandas.DataFrame`.
    """
    df = pd.read_csv(input_path)

    # --- Validate required columns ----------------------------------------
    required = {
        "determinand.prefLabel",
        "result",
        "samplingPoint.easting",
        "samplingPoint.northing",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # --- Add latitude / longitude -----------------------------------------
    eastings = np.asarray(df["samplingPoint.easting"].values, dtype=float)
    northings = np.asarray(df["samplingPoint.northing"].values, dtype=float)
    lat, lon = easting_northing_to_latlon(eastings, northings)
    df["latitude"] = lat
    df["longitude"] = lon

    # --- Rename / pivot the result column ---------------------------------
    unique_determinands = df["determinand.prefLabel"].unique()

    if len(unique_determinands) == 1:
        # Simple case – just rename the column
        determinand_name = unique_determinands[0]
        unit = df["unit"].iloc[0] if "unit" in df.columns else None
        col_name = _build_column_name(determinand_name, unit)
        df = df.rename(columns={"result": col_name})
        df = df.drop(
            columns=["determinand.prefLabel", "determinand.notation", "unit"], errors="ignore"
        )
    else:
        # Multiple determinands – pivot so each gets its own column.
        # Build a human-friendly column name per determinand.
        det_unit_map: dict[str, str] = {}
        if "unit" in df.columns:
            for det in unique_determinands:
                unit = df.loc[df["determinand.prefLabel"] == det, "unit"].iloc[0]
                det_unit_map[det] = _build_column_name(det, unit)
        else:
            det_unit_map = {d: d for d in unique_determinands}

        df["_det_col"] = df["determinand.prefLabel"].map(lambda d: det_unit_map[d])

        # Identify which id columns are actually present in the dataframe
        index_cols = [c for c in _ID_COLS if c in df.columns] + ["latitude", "longitude"]

        pivot = df.pivot_table(
            index=index_cols,
            columns="_det_col",
            values="result",
            aggfunc="first",
        ).reset_index()

        # Flatten the column index
        pivot.columns.name = None
        df = pivot

    # --- Write output -----------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Converted CSV written to %s  (%d rows)", output_path, len(df))
    return df


# ---- CLI ------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raw EA water-quality CSV into a cleaner format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("ea_script_out.csv"),
        help="Path to the raw EA CSV file.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("ea_water_quality_clean.csv"),
        help="Destination path for the converted CSV.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    try:
        convert(args.input, args.output)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
