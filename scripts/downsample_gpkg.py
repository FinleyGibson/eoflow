"""
Downsample a delineated-catchments GeoPackage (from ``delineate_catchments.py``)
to a target number of samples, maximising diversity across sample sites and
sample times.

Selection strategy
-------------------
1. **Site diversity.** If ``n_samples >= n_sites``, every site keeps at least
   one sample. If ``n_samples < n_sites``, sites are chosen via farthest-point
   sampling on their location, so the reduced site set stays spatially spread
   out rather than clustered.
2. **Slot apportionment.** Once every site has its guaranteed sample, any
   remaining budget is distributed across sites via a capped divisor
   (D'Hondt-style) method weighted by ``site_sample_count ** diversity_power``
   — sites with more history get proportionally more slots, but the exponent
   (default 0.5, i.e. sqrt) tempers how much a handful of heavily-sampled
   sites can dominate the budget.
3. **Time diversity.** Within a site's quota, samples are picked evenly
   spaced across that site's sorted date range (not randomly), so both the
   earliest and latest visits stay represented rather than a random cluster.

The whole procedure is deterministic — the same input and ``--n-samples``
always produce the same output.

Usage
-----
    python -m scripts.downsample_gpkg \\
        --gpkg data/delineated_catchments/devon_turbidity_sites_2010-10_2026-06.pkg \\
        --n-samples 1000 \\
        --output data/delineated_catchments/devon_turbidity_sites_1000.gpkg
"""

from __future__ import annotations

import argparse
import heapq
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np

from eoflow.log_utils import get_logger
from eoflow.samples import CatchmentDataset, Sample

logger = get_logger(__file__)


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def _evenly_spaced_indices(n_available: int, k: int) -> List[int]:
    """*k* distinct, sorted positions in ``range(n_available)``, evenly spread.

    Endpoints are included when ``k >= 2`` so both extremes of whatever
    *n_available* items are ordered by stay represented.
    """
    if k <= 0:
        return []
    if k >= n_available:
        return list(range(n_available))

    raw = np.round(np.linspace(0, n_available - 1, k)).astype(int)
    seen: set[int] = set()
    for i in raw:
        i = int(i)
        while i in seen and i < n_available - 1:
            i += 1
        while i in seen and i > 0:
            i -= 1
        seen.add(i)
    return sorted(seen)


def _apportion_extra_slots(
    counts: Dict[str, int],
    remaining: int,
    *,
    diversity_power: float,
) -> Dict[str, int]:
    """Distribute *remaining* slots across sites beyond their guaranteed 1.

    Uses a capped divisor (D'Hondt-style) method: each site's priority for
    its next slot is ``weight / (current_alloc + 1)``, where
    ``weight = count ** diversity_power``. A site's total allocation can
    never exceed its own sample count.
    """
    alloc = {site: 1 for site in counts}
    weight = {site: count**diversity_power for site, count in counts.items()}

    heap = [(-weight[s] / (alloc[s] + 1), s) for s in counts if alloc[s] < counts[s]]
    heapq.heapify(heap)

    while remaining > 0 and heap:
        _, s = heapq.heappop(heap)
        alloc[s] += 1
        remaining -= 1
        if alloc[s] < counts[s]:
            heapq.heappush(heap, (-weight[s] / (alloc[s] + 1), s))

    return alloc


def _select_sites_fps(site_locations: Dict[str, Tuple[float, float]], n_wanted: int) -> List[str]:
    """Farthest-point sampling: pick *n_wanted* sites spread out spatially."""
    sites = sorted(site_locations)
    chosen = [sites[0]]
    remaining = set(sites[1:])

    while len(chosen) < n_wanted and remaining:
        best_site, best_dist = None, -1.0
        for s in remaining:
            sx, sy = site_locations[s]
            d = min((sx - site_locations[c][0]) ** 2 + (sy - site_locations[c][1]) ** 2 for c in chosen)
            if d > best_dist:
                best_dist, best_site = d, s
        chosen.append(best_site)
        remaining.discard(best_site)

    return chosen


def _sort_key(sample: Sample) -> Tuple[bool, date]:
    """Sort samples chronologically, pushing dateless ones to the end."""
    return (sample.date is None, sample.date or date.max)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def downsample(
    gpkg_path: Path,
    n_samples: int,
    output_path: Path,
    *,
    only_delineated: bool = True,
    diversity_power: float = 0.5,
) -> gpd.GeoDataFrame:
    """Downsample *gpkg_path* to *n_samples* rows, maximising site/time diversity.

    Parameters
    ----------
    gpkg_path : Path
        GeoPackage produced by ``delineate_catchments.py``.
    n_samples : int
        Target number of samples to keep.
    output_path : Path
        Destination GeoPackage.
    only_delineated : bool
        If *True* (default), only rows with a successfully delineated
        catchment are eligible for selection.
    diversity_power : float
        Exponent applied to each site's sample count when apportioning slots
        beyond the 1-per-site guarantee. ``1.0`` allocates proportionally to
        how much history a site has; ``0.0`` allocates equally regardless of
        it; the default ``0.5`` (sqrt) favours diversity over raw density.

    Returns
    -------
    geopandas.GeoDataFrame
        The selected subset, also written to *output_path*.
    """
    dataset = CatchmentDataset.from_gpkg(gpkg_path, only_delineated=only_delineated)
    samples = list(dataset)
    total = len(samples)
    logger.info("Loaded %d eligible sample(s) from %s", total, gpkg_path)

    by_site: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        by_site.setdefault(s.notation, []).append(i)
    n_sites = len(by_site)

    if n_samples >= total:
        logger.warning(
            "Requested n_samples=%d >= available %d; keeping all samples", n_samples, total
        )
        selected_idx = list(range(total))
    elif n_samples >= n_sites:
        counts = {site: len(idxs) for site, idxs in by_site.items()}
        alloc = _apportion_extra_slots(counts, n_samples - n_sites, diversity_power=diversity_power)
        selected_idx = []
        for site, idxs in by_site.items():
            idxs_sorted = sorted(idxs, key=lambda i: _sort_key(samples[i]))
            positions = _evenly_spaced_indices(len(idxs_sorted), alloc[site])
            selected_idx.extend(idxs_sorted[p] for p in positions)
    else:
        site_locations = {
            site: (samples[idxs[0]].latitude or 0.0, samples[idxs[0]].longitude or 0.0)
            for site, idxs in by_site.items()
        }
        chosen_sites = _select_sites_fps(site_locations, n_samples)
        selected_idx = []
        for site in chosen_sites:
            idxs_sorted = sorted(by_site[site], key=lambda i: _sort_key(samples[i]))
            positions = _evenly_spaced_indices(len(idxs_sorted), 1)
            selected_idx.extend(idxs_sorted[p] for p in positions)

    selected_idx = sorted(set(selected_idx))
    gdf_selected = dataset.to_geodataframe().iloc[selected_idx].reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_selected.to_file(str(output_path), driver="GPKG")

    sites_selected = {samples[i].notation for i in selected_idx}
    dated = [samples[i].date for i in selected_idx if samples[i].date is not None]
    date_range = f", date range {min(dated)} to {max(dated)}" if dated else ""
    logger.info(
        "Selected %d/%d sample(s) across %d/%d site(s)%s; wrote %s",
        len(selected_idx),
        total,
        len(sites_selected),
        n_sites,
        date_range,
        output_path,
    )
    return gdf_selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Downsample a delineated-catchments GeoPackage to a target number "
            "of samples, maximising diversity across sample sites and times."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gpkg",
        type=Path,
        required=True,
        help="Path to the delineated-catchments GeoPackage (from delineate_catchments.py).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Target number of samples to keep.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination GeoPackage for the downsampled subset.",
    )
    p.add_argument(
        "--include-undelineated",
        dest="only_delineated",
        action="store_false",
        default=True,
        help="Include rows without a successfully delineated catchment (excluded by default).",
    )
    p.add_argument(
        "--diversity-power",
        type=float,
        default=0.5,
        help=(
            "Exponent applied to each site's sample count when apportioning slots "
            "beyond the 1-per-site guarantee. 1.0=proportional to density, "
            "0.0=equal regardless of density, default 0.5=sqrt (favours diversity)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.gpkg.exists():
        logger.error("GeoPackage not found: %s", args.gpkg)
        sys.exit(1)
    if args.n_samples <= 0:
        logger.error("--n-samples must be positive, got %d", args.n_samples)
        sys.exit(1)

    try:
        downsample(
            gpkg_path=args.gpkg,
            n_samples=args.n_samples,
            output_path=args.output,
            only_delineated=args.only_delineated,
            diversity_power=args.diversity_power,
        )
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
