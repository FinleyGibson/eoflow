"""
scripts/prepare_samples.py
--------------------------
Load a GeoPackage of delineated catchments, compute spatial layers
(topography, slope, aspect, soil, rainfall) and fetch Earth-observation
indices (NDVI, NDWI) via openEO for each sample, then save each
fully-populated Sample to disk so that ``extract_features.py`` can
read them.

Usage
-----
  # Defaults: reads outputs/devon_catchments.gpkg -> data/sample_instances/
  python -m scripts.prepare_samples

  # Custom paths
  python -m scripts.prepare_samples \
      --gpkg    outputs/devon_catchments.gpkg \
      --dem     data/dems/devon_dem_cop30.tif \
      --out-dir data/sample_instances \
      --rainfall-start 2023-01-01 \
      --rainfall-end   2023-12-31 \
      --eo-start       2023-01-01 \
      --eo-end         2023-12-31

  # Skip the EO fetch step
  python -m scripts.prepare_samples --skip-eo
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from eoflow.log_utils import get_logger
from eoflow.utils import DATA_DIR, PROJECT_ROOT

logger = get_logger("eoflow.scripts.prepare_samples")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_GPKG: Path = PROJECT_ROOT / "outputs" / "devon_catchments.gpkg"
DEFAULT_DEM: Path = DATA_DIR / "dems" / "devon_dem_cop30.tif"
DEFAULT_OUT_DIR: Path = DATA_DIR / "sample_instances"
DEFAULT_RAINFALL_DIR: Path = DATA_DIR / "temp" / "rainfall"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Prepare saved Sample instances from a GeoPackage of delineated "
            "catchments.  Computes spatial layers and fetches EO indices for "
            "each sample, then saves to disk for use by extract_features.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gpkg",
        type=Path,
        default=DEFAULT_GPKG,
        metavar="FILE",
        help="Input GeoPackage produced by dataset_builder.py.",
    )
    p.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM,
        metavar="FILE",
        help="DEM GeoTIFF for topography / slope / aspect computation.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help="Output directory for saved Sample subdirectories.",
    )
    p.add_argument(
        "--rainfall-dir",
        type=Path,
        default=DEFAULT_RAINFALL_DIR,
        metavar="DIR",
        help="Cache directory for Met Office rainfall NetCDF downloads.",
    )
    p.add_argument(
        "--rainfall-start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date for rainfall extraction.  Omit to skip rainfall.",
    )
    p.add_argument(
        "--rainfall-end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="End date for rainfall extraction.  Omit to skip rainfall.",
    )
    p.add_argument(
        "--eo-start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date for NDVI/NDWI EO queries.  Omit to use per-sample defaults.",
    )
    p.add_argument(
        "--eo-end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="End date for NDVI/NDWI EO queries.  Omit to use per-sample defaults.",
    )
    p.add_argument(
        "--skip-eo",
        action="store_true",
        help="Skip the openEO NDVI/NDWI fetch step entirely.",
    )
    p.add_argument(
        "--flow-acc-threshold",
        type=int,
        default=5000,
        metavar="N",
        help="Flow-accumulation threshold for pour-point snapping during layer computation.",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(s):
    """Parse a YYYY-MM-DD string into a datetime, or return None."""
    if s is None:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError as exc:
        logger.error("Invalid date format '%s': %s", s, exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Lazy imports so --help is fast
    from eoflow.samples import CatchmentDataset

    logger.info("=" * 60)
    logger.info("Prepare sample instances")
    logger.info("  gpkg        : %s", args.gpkg)
    logger.info("  dem         : %s", args.dem)
    logger.info("  out-dir     : %s", args.out_dir)
    logger.info("  rainfall    : %s -> %s", args.rainfall_start, args.rainfall_end)
    logger.info("  EO window   : %s -> %s", args.eo_start, args.eo_end)
    logger.info("  skip EO     : %s", args.skip_eo)
    logger.info("=" * 60)

    # -- Validate inputs ---------------------------------------------------
    if not args.gpkg.exists():
        logger.error("GeoPackage not found: %s", args.gpkg)
        sys.exit(1)
    if not args.dem.exists():
        logger.error("DEM not found: %s", args.dem)
        sys.exit(1)

    rainfall_start = _parse_date(args.rainfall_start)
    rainfall_end = _parse_date(args.rainfall_end)

    # -- Load GeoPackage ---------------------------------------------------
    ds = CatchmentDataset.from_gpkg(args.gpkg, only_delineated=True)
    logger.info("Loaded %d delineated samples from %s", len(ds), args.gpkg)

    if len(ds) == 0:
        logger.error("No delineated samples found -- nothing to do.")
        sys.exit(1)

    # -- Connect to openEO (once, reused for all samples) ------------------
    conn = None
    if not args.skip_eo:
        try:
            logger.info("Connecting to openEO ...")
            conn = Sample.connect_openeo()
            logger.info("openEO connection established.")
        except Exception as exc:
            logger.warning("Could not connect to openEO -- EO fetch will be skipped: %s", exc)

    # -- Process each sample -----------------------------------------------
    n_total = len(ds)
    n_saved = 0
    n_skipped = 0
    n_failed = 0

    for i, sample in enumerate(ds):
        tag = sample.notation or "sample_{:04d}".format(i)
        out_dir = args.out_dir / tag

        # Skip if already saved
        if (out_dir / "metadata.json").exists():
            logger.info("  [%d/%d] skip %s (already exists)", i + 1, n_total, tag)
            n_skipped += 1
            continue

        logger.info("  [%d/%d] processing %s ...", i + 1, n_total, tag)

        try:
            # Compute terrain + soil + rainfall layers
            sample.compute_layers(
                dem_path=args.dem,
                rainfall_start=rainfall_start,
                rainfall_end=rainfall_end,
                rainfall_download_dir=args.rainfall_dir,
                flow_acc_threshold=args.flow_acc_threshold,
            )

            # Fetch NDVI and NDWI from openEO / Sentinel-2
            if conn is not None:
                try:
                    sample.fetch_ndvi(
                        conn,
                        start_date=args.eo_start,
                        end_date=args.eo_end,
                    )
                    sample.fetch_ndwi(
                        conn,
                        start_date=args.eo_start,
                        end_date=args.eo_end,
                    )
                except Exception as exc:
                    logger.warning("  EO fetch failed for %s: %s", tag, exc)

            # Save the fully-populated Sample to disk
            sample.save(out_dir)
            n_saved += 1
            logger.info("  saved %s", tag)

        except Exception as exc:
            n_failed += 1
            logger.warning("  %s failed: %s", tag, exc)

    # -- Summary -----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Preparation complete.")
    logger.info("  Saved   : %d", n_saved)
    logger.info("  Skipped : %d (already existed)", n_skipped)
    logger.info("  Failed  : %d", n_failed)
    logger.info("  Output  : %s", args.out_dir)
    logger.info("=" * 60)

    if n_saved == 0 and n_skipped == 0:
        logger.error("No samples were saved -- check the logs above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
