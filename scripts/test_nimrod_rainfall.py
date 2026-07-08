"""
Quick smoke-test for the NIMROD disk-based rainfall loading.

Loads a pre-saved Sample and calls ``compute_rainfall`` over a date range
using the local NIMROD data in ``data/nimrod_data/``.

Usage
-----
::

    python -m scripts.test_nimrod_rainfall 1
    python -m scripts.test_nimrod_rainfall 42

The numeric argument selects the source sample (``data/sample_instances/sample_001``
etc.) and names the output directory (``data/sample_instances/sample_001_nimrod_test``).

Run from the repository root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eoflow.log_utils import get_logger
from eoflow.samples import Sample

logger = get_logger(__file__)

# ---------------------------------------------------------------------------
# Fixed paths and time window
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path("data/sample_instances/sample_00")
NIMROD_DIR = Path("data/nimrod_data")
OUT_BASE = Path("data/sample_instances")

# A window that falls within the 2023-01-06 data (12:00–23:55 UTC, 5-min steps)
START = "2023-01-06T12:00"
END = "2023-10-06T13:00"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-test NIMROD rainfall loading for a given sample index."
    )
    p.add_argument(
        "index",
        type=int,
        help="Sample index (e.g. 1 → sample_001).  Zero-padded to three digits.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = OUT_BASE / f"sample_{args.index:03d}_nimrod_test"

    # --- Load the pre-saved sample -------------------------------------------
    print(f"Loading sample from: {SAMPLE_DIR}")
    sample = Sample.load(SAMPLE_DIR)
    print(f"  Site      : {sample.site_name}")
    print(f"  Notation  : {sample.notation}")
    print(f"  Catchment : {'yes' if sample.has_catchment else 'NO — test will fail'}")
    print()

    # --- Load rainfall from disk ---------------------------------------------
    print(f"Loading NIMROD rainfall for {START} → {END}")
    print(f"  Source    : {NIMROD_DIR}")
    da = sample.compute_rainfall(START, END, nimrod_dir=NIMROD_DIR)
    print()

    # --- Report results -------------------------------------------------------
    if da.size == 0:
        logger.warning("Returned an empty DataArray — no files matched the time window.")
        return

    print("Result:")
    print(f"  Shape     : {dict(da.sizes)}")
    print(f"  Dims      : {da.dims}")
    print(f"  CRS       : {da.attrs.get('crs', 'n/a')}")
    print(f"  Units     : {da.attrs.get('units', 'n/a')}")
    print(f"  Source    : {da.attrs.get('source', 'n/a')}")
    print(f"  Time range: {str(da.time.values[0])[:19]}  →  {str(da.time.values[-1])[:19]}")
    print(f"  Min (mm/h): {float(da.min(skipna=True)):.4f}")
    print(f"  Max (mm/h): {float(da.max(skipna=True)):.4f}")
    print(f"  Mean(mm/h): {float(da.mean(skipna=True)):.4f}")
    print()
    print("layers.rainfall set:", sample.layers.rainfall is not None)

    # --- Save to disk for interactive inspection -----------------------------
    print()
    print(f"Saving sample to: {out_dir}")
    sample.save(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
