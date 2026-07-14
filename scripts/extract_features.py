"""
scripts/extract_features.py
────────────────────────────
Apply feature extraction to a directory of saved Sample instances and
write the resulting DataFrame to disk as a CSV (and optionally as a
Parquet file).

Usage
─────
  # Extract from the default samples dir → outputs/catchment_features.csv
  python scripts/extract_features.py

  # Custom samples dir and output path
  python scripts/extract_features.py \
      --samples-dir data/sample_instances \
      --output      outputs/my_features.csv

  # Also save a Parquet file (preserves types better than CSV)
  python scripts/extract_features.py --parquet

  # Adjust NDVI bare-earth threshold
  python scripts/extract_features.py --bare-ndvi 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eoflow.features import extract_features_batch, print_feature_summary
from eoflow.log_utils import get_logger
from eoflow.utils import DATA_DIR, PROJECT_ROOT

logger = get_logger(__file__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SAMPLES_DIR: Path = DATA_DIR / "sample_instances"
DEFAULT_OUTPUT_CSV: Path = PROJECT_ROOT / "outputs" / "catchment_features.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract catchment-scale regression features from saved Samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        metavar="DIR",
        help="Directory containing saved Sample subdirectories.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        metavar="FILE",
        help="Destination CSV path.",
    )
    p.add_argument(
        "--parquet",
        action="store_true",
        help="Also save a Parquet file alongside the CSV (same stem, .parquet extension).",
    )
    # Threshold overrides
    p.add_argument(
        "--bare-ndvi",
        type=float,
        default=0.20,
        metavar="THRESH",
        help="NDVI threshold below which a pixel is classified as bare/sparse cover.",
    )
    p.add_argument(
        "--dense-ndvi",
        type=float,
        default=0.60,
        metavar="THRESH",
        help="NDVI threshold above which a pixel is classified as dense vegetation.",
    )
    p.add_argument(
        "--wet-ndwi",
        type=float,
        default=0.00,
        metavar="THRESH",
        help="NDWI threshold above which a pixel is classified as open water / saturated.",
    )
    p.add_argument(
        "--steep-low",
        type=float,
        default=10.0,
        metavar="DEG",
        help="Primary steep-slope threshold (degrees).",
    )
    p.add_argument(
        "--steep-high",
        type=float,
        default=15.0,
        metavar="DEG",
        help="Secondary steep-slope threshold (degrees).",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    logger.info("=" * 60)
    logger.info("Feature extraction")
    logger.info("  samples dir : %s", args.samples_dir)
    logger.info("  output      : %s", args.output)
    logger.info("  bare NDVI   : %.2f", args.bare_ndvi)
    logger.info("  dense NDVI  : %.2f", args.dense_ndvi)
    logger.info("  wet NDWI    : %.2f", args.wet_ndwi)
    logger.info("  steep low   : %.1f°", args.steep_low)
    logger.info("  steep high  : %.1f°", args.steep_high)
    logger.info("=" * 60)

    # ── Extract ──────────────────────────────────────────────────────────
    df = extract_features_batch(
        args.samples_dir,
        bare_ndvi_threshold=args.bare_ndvi,
        dense_ndvi_threshold=args.dense_ndvi,
        wet_ndwi_threshold=args.wet_ndwi,
        steep_slope_low=args.steep_low,
        steep_slope_high=args.steep_high,
    )

    # ── Save CSV ─────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved CSV  → %s  (%d rows × %d cols)", args.output, *df.shape)

    # ── Optional Parquet ─────────────────────────────────────────────────
    if args.parquet:
        parquet_path = args.output.with_suffix(".parquet")
        try:
            df.to_parquet(parquet_path, index=False)
            logger.info("Saved Parquet → %s", parquet_path)
        except ImportError:
            logger.warning(
                "pyarrow/fastparquet not installed — Parquet skipped. pip install pyarrow"
            )

    # ── Summary report ───────────────────────────────────────────────────
    print_feature_summary(df, args.output)


if __name__ == "__main__":
    main()
