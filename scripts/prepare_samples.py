"""
scripts/prepare_samples.py
--------------------------
Load a GeoPackage of delineated catchments, compute spatial layers
(topography, slope, aspect, soil, rainfall) and fetch Earth-observation
indices (NDVI, NDWI) via openEO for each sample, then save each
fully-populated Sample to disk so that ``extract_features.py`` can
read them.

Rainfall and EO date windows are specified as a number of days *before*
each sample's own measurement date, so the window is per-sample rather
than a single fixed range.

Usage
-----
  # Defaults: reads outputs/devon_catchments.gpkg -> data/sample_instances/
  python -m scripts.prepare_samples

  # Custom paths, 10-day rainfall window, 30-day EO window
  python -m scripts.prepare_samples \
      --gpkg          outputs/devon_catchments.gpkg \
      --dem           data/dems/devon_dem_cop30.tif \
      --out-dir       data/sample_instances \
      --nimrod-dir    data/nimrod_data \
      --rainfall-days 10 \
      --eo-days       30

  # Skip the EO fetch step
  python -m scripts.prepare_samples --skip-eo
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

from eoflow.log_utils import get_logger, set_level
from eoflow.utils import DATA_DIR, PROJECT_ROOT

logger = get_logger(__file__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_GPKG: Path = PROJECT_ROOT / "outputs" / "devon_catchments.gpkg"
DEFAULT_DEM: Path = DATA_DIR / "dems" / "devon_dem_cop30.tif"
DEFAULT_OUT_DIR: Path = DATA_DIR / "sample_instances"
DEFAULT_NIMROD_DIR: Path = DATA_DIR / "nimrod_data"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
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
        "--nimrod-dir",
        type=Path,
        default=DEFAULT_NIMROD_DIR,
        metavar="DIR",
        help=(
            "Path to NIMROD data: either a directory of per-timestep NetCDF files "
            "(as produced by process_nimrod_local.py) or a single consolidated .nc file. "
            "Omit --rainfall-days to skip rainfall."
        ),
    )
    p.add_argument(
        "--rainfall-days",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of days before each sample's measurement date to use as the "
            "rainfall window.  Omit to skip rainfall computation."
        ),
    )
    p.add_argument(
        "--eo-days",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of days before each sample's measurement date to use as the "
            "Sentinel-2 EO window.  Omit to use the openEO per-sample default."
        ),
    )
    p.add_argument(
        "--skip-eo",
        action="store_true",
        help="Skip the openEO NDVI/NDWI fetch step entirely.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
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


def _window_for_sample(
    sample_date: Optional[date],
    days: int,
    label: str,
    tag: str,
) -> Optional[Tuple[str, str]]:
    """Return (start_str, end_str) for a days-before window, or None on failure."""
    if sample_date is None:
        logger.warning("  %s: no sample date — skipping %s window", tag, label)
        return None
    end = sample_date
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    # Lazy imports so --help is fast
    from eoflow.samples import CatchmentDataset, Sample

    set_level(logger, args.log_level)

    logger.info("=" * 60)
    logger.info("Prepare sample instances")
    logger.info("  gpkg           : %s", args.gpkg)
    logger.info("  dem            : %s", args.dem)
    logger.info("  out-dir        : %s", args.out_dir)
    logger.info("  nimrod-dir     : %s", args.nimrod_dir)
    logger.info("  rainfall-days  : %s", args.rainfall_days)
    logger.info("  eo-days        : %s", args.eo_days)
    logger.info("  skip EO        : %s", args.skip_eo)
    logger.info("=" * 60)

    # -- Validate inputs ---------------------------------------------------
    if not args.gpkg.exists():
        logger.error("GeoPackage not found: %s", args.gpkg)
        sys.exit(1)
    if not args.dem.exists():
        logger.error("DEM not found: %s", args.dem)
        sys.exit(1)
    if args.rainfall_days is not None and not args.nimrod_dir.exists():
        logger.error("NIMROD directory not found: %s", args.nimrod_dir)
        sys.exit(1)

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

        logger.info("  [%d/%d] processing %s (date: %s) ...", i + 1, n_total, tag, sample.date)

        # Derive per-sample date windows
        rainfall_window = None
        if args.rainfall_days is not None:
            rainfall_window = _window_for_sample(sample.date, args.rainfall_days, "rainfall", tag)
            if rainfall_window:
                logger.info(
                    "    rainfall window : %s → %s (%d days)",
                    *rainfall_window,
                    args.rainfall_days,
                )

        eo_window = None
        if args.eo_days is not None:
            eo_window = _window_for_sample(sample.date, args.eo_days, "EO", tag)
            if eo_window:
                logger.info(
                    "    EO window       : %s → %s (%d days)",
                    *eo_window,
                    args.eo_days,
                )

        try:
            # Compute terrain + soil + rainfall layers
            sample.compute_layers(
                dem_path=args.dem,
                rainfall_start=rainfall_window[0] if rainfall_window else None,
                rainfall_end=rainfall_window[1] if rainfall_window else None,
                rainfall_nimrod_dir=args.nimrod_dir if rainfall_window else None,
                flow_acc_threshold=args.flow_acc_threshold,
            )

            # Fetch NDVI and NDWI from openEO / Sentinel-2
            if conn is not None:
                eo_start = eo_window[0] if eo_window else None
                eo_end = eo_window[1] if eo_window else None
                try:
                    sample.fetch_ndvi(conn, start_date=eo_start, end_date=eo_end)
                    sample.fetch_ndwi(conn, start_date=eo_start, end_date=eo_end)
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
