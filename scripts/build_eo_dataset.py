"""
Build a per-sample EO feature table by indexing the whole-region, per-year
datacubes produced by ``scripts.fetch_annual_eo_cubes`` with each sample's
own delineated catchment polygon.

For every sample in a :class:`~eoflow.samples.CatchmentDataset` GeoPackage
(from ``delineate_catchments.py``) that has a catchment polygon and a date,
this script:

1. Picks a ``sample.date ± days_window`` time window.
2. Opens whichever year(s) of pre-fetched annual cube(s) overlap that
   window (:mod:`scripts.fetch_annual_eo_cubes` output).
3. Masks each cube to the sample's catchment polygon (reprojected to the
   cube's CRS — Sentinel-2 UTM zone, EPSG:32630 for Devon) and computes the
   mean/std of each requested band/index over the window.

The result is written as a single CSV with one row per sample: identity
columns (notation, site name, date, result, unit, catchment area) plus
``<band>_mean``/``<band>_std`` and ``<index>_mean``/``<index>_std`` columns.

Bands, indices, and the day window all default to the ``eo`` section of
:mod:`eoflow.config` and can be overridden per-invocation via CLI flags.

Usage
-----
    python -m scripts.build_eo_dataset \\
        --gpkg data/delineated_catchments/devon_turbidity_sites_2010-10_2026-06.pkg \\
        --cubes-dir data/eo_cubes/devon_annual \\
        --output data/eo_datasets/devon_turbidity_features.csv

Notes
-----
* Requires the optional ``xarray`` dependency (see :mod:`eoflow.eo`).
* Run ``scripts.fetch_annual_eo_cubes`` first to produce the per-year
  NetCDF cubes this script reads.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform
import xarray as xr
from shapely.geometry.base import BaseGeometry

from eoflow.config import get_config
from eoflow.log_utils import get_logger
from eoflow.samples import CatchmentDataset

logger = get_logger(__file__)

#: CRS the CDSE openEO backend returns Sentinel-2 datacubes in (matches the
#: convention already used for per-sample EO layers in eoflow.features).
_EO_CRS = "EPSG:32630"

#: Valid-value ranges used to reject obviously corrupt pixels before
#: averaging. Spectral indices are physically bounded to [-1, 1]; raw
#: reflectance bands have no such bound, so they are left unclipped.
_INDEX_VALUE_RANGE: Tuple[float, float] = (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Catchment masking (mirrors eoflow.features._eo_catchment_mask)
# ---------------------------------------------------------------------------


def _catchment_mask(
    catchment_geom: BaseGeometry,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """Boolean mask (True = inside catchment) for a grid in ``_EO_CRS``."""
    catchment_utm = gpd.GeoSeries([catchment_geom], crs="EPSG:4326").to_crs(_EO_CRS).iloc[0]

    x_c = np.asarray(x_coords, dtype=float)
    y_c = np.asarray(y_coords, dtype=float)
    dx = x_c[1] - x_c[0]
    dy = y_c[1] - y_c[0]  # negative for north-up rasters

    affine = rasterio.transform.from_origin(
        x_c[0] - abs(dx) / 2.0,
        y_c[0] + abs(dy) / 2.0,
        abs(dx),
        abs(dy),
    )
    outside = rasterio.features.geometry_mask(
        [catchment_utm.__geo_interface__],
        out_shape=(len(y_c), len(x_c)),
        transform=affine,
        invert=False,
    )
    return ~outside


def _masked_stats(
    da: "xr.DataArray",
    inside: np.ndarray,
    *,
    value_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Mean/std of *da* (dims ``t, y, x``) over time, masked to *inside*."""
    vals = da.values.astype(float)
    if value_range is not None:
        vmin, vmax = value_range
        vals = np.where((vals >= vmin) & (vals <= vmax), vals, np.nan)
    vals[:, ~inside] = np.nan

    with np.errstate(all="ignore"):
        finite = vals[np.isfinite(vals)]

    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


# ---------------------------------------------------------------------------
# Annual cube loading
# ---------------------------------------------------------------------------


class _AnnualCubeCache:
    """Lazily opens and caches per-year NetCDF cubes from *cubes_dir*."""

    def __init__(self, cubes_dir: Path) -> None:
        self.cubes_dir = Path(cubes_dir)
        self._cache: Dict[Tuple[int, str], Optional["xr.Dataset"]] = {}

    def _get(self, year: int, filename: str) -> Optional["xr.Dataset"]:
        key = (year, filename)
        if key not in self._cache:
            path = self.cubes_dir / str(year) / filename
            if path.exists():
                self._cache[key] = xr.open_dataset(path)
            else:
                logger.warning("Missing cube for year %d: %s", year, path)
                self._cache[key] = None
        return self._cache[key]

    def window(self, filename: str, start: date, end: date) -> Optional["xr.Dataset"]:
        """Return the time slice [*start*, *end*] across whichever year file(s) exist."""
        parts = []
        for year in range(start.year, end.year + 1):
            ds = self._get(year, filename)
            if ds is None:
                continue
            parts.append(ds.sel(t=slice(start.isoformat(), end.isoformat())))

        if not parts:
            return None
        return parts[0] if len(parts) == 1 else xr.concat(parts, dim="t")

    def close(self) -> None:
        for ds in self._cache.values():
            if ds is not None:
                ds.close()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def build_dataset(
    gpkg_path: Path,
    cubes_dir: Path,
    output_path: Path,
    *,
    bands: Optional[Sequence[str]] = None,
    indices: Optional[Sequence[str]] = None,
    days_window: Optional[int] = None,
) -> pd.DataFrame:
    """Build a per-sample EO feature table from pre-fetched annual cubes.

    Parameters
    ----------
    gpkg_path : Path
        GeoPackage produced by ``delineate_catchments.py``.
    cubes_dir : Path
        Root directory of per-year cubes written by
        ``scripts.fetch_annual_eo_cubes`` (``<year>/bands.nc``,
        ``<year>/{index}.nc``).
    output_path : Path
        Destination CSV file.
    bands : sequence of str, optional
        Band names to extract from ``bands.nc``. Defaults to the
        ``eo.bands`` config value.
    indices : sequence of str, optional
        Index names to extract (one file per index). Defaults to the
        ``eo.indices`` config value.
    days_window : int, optional
        Half-width in days of the averaging window around each sample's
        date. Defaults to the ``eo.days_window`` config value.

    Returns
    -------
    pandas.DataFrame
        The feature table, also written to *output_path*.
    """
    config = get_config()
    bands = list(bands) if bands is not None else list(config.get("eo.bands", []))
    indices = list(indices) if indices is not None else list(config.get("eo.indices", []))
    days_window = days_window if days_window is not None else config.get("eo.days_window", 15)

    logger.info("Loading catchment dataset from %s …", gpkg_path)
    dataset = CatchmentDataset.from_gpkg(gpkg_path, only_delineated=True)
    logger.info("Loaded %d delineated sample(s)", len(dataset))

    cache = _AnnualCubeCache(cubes_dir)
    window = timedelta(days=days_window)
    records: List[dict] = []

    try:
        for sample in dataset:
            row: dict = {
                "notation": sample.notation,
                "site_name": sample.site_name,
                "date": sample.date,
                "result": sample.result,
                "unit": sample.unit,
                "catchment_area_km2": sample.catchment_area_km2(),
            }

            feature_names = list(bands) + [i.lower() for i in indices]
            if not sample.has_catchment or sample.date is None:
                logger.warning(
                    "Skipping EO extraction for '%s': no catchment/date",
                    sample.notation,
                )
                for name in feature_names:
                    row[f"{name}_mean"] = np.nan
                    row[f"{name}_std"] = np.nan
                records.append(row)
                continue

            start, end = sample.date - window, sample.date + window

            if bands:
                ds = cache.window("bands.nc", start, end)
                inside = (
                    _catchment_mask(sample.catchment, ds.coords["x"].values, ds.coords["y"].values)
                    if ds is not None and ds.sizes.get("t", 0) > 0
                    else None
                )
                for band in bands:
                    if inside is not None and band in ds.data_vars:
                        mean, std = _masked_stats(ds[band], inside)
                    else:
                        mean, std = float("nan"), float("nan")
                    row[f"{band}_mean"] = mean
                    row[f"{band}_std"] = std

            for index in indices:
                var_name = index.lower()
                ds = cache.window(f"{var_name}.nc", start, end)
                if ds is not None and ds.sizes.get("t", 0) > 0:
                    da = ds[var_name] if var_name in ds.data_vars else next(iter(ds.data_vars.values()))
                    inside = _catchment_mask(sample.catchment, ds.coords["x"].values, ds.coords["y"].values)
                    mean, std = _masked_stats(da, inside, value_range=_INDEX_VALUE_RANGE)
                else:
                    mean, std = float("nan"), float("nan")
                row[f"{var_name}_mean"] = mean
                row[f"{var_name}_std"] = std

            records.append(row)
    finally:
        cache.close()

    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote EO feature dataset (%d rows) to %s", len(df), output_path)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from eoflow import eo

    config = get_config()
    p = argparse.ArgumentParser(
        description=(
            "Build a per-sample EO feature table by indexing pre-fetched "
            "annual datacubes with each sample's catchment polygon."
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
        "--cubes-dir",
        type=Path,
        required=True,
        help="Root directory of per-year cubes written by scripts.fetch_annual_eo_cubes.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV file for the feature table.",
    )
    p.add_argument(
        "--bands",
        nargs="+",
        default=None,
        metavar="BAND",
        help=f"Band names to extract. Defaults to config eo.bands ({config.get('eo.bands')}).",
    )
    p.add_argument(
        "--indices",
        nargs="+",
        default=None,
        metavar="INDEX",
        type=str.upper,
        choices=sorted(eo.SUPPORTED_INDICES),
        help=f"Index names to extract. Defaults to config eo.indices ({config.get('eo.indices')}).",
    )
    p.add_argument(
        "--days-window",
        type=int,
        default=None,
        help=(
            "Half-width in days of the averaging window around each sample's "
            f"date. Defaults to config eo.days_window ({config.get('eo.days_window')})."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.gpkg.exists():
        logger.error("GeoPackage not found: %s", args.gpkg)
        sys.exit(1)

    if not args.cubes_dir.exists():
        logger.error("Cubes directory not found: %s", args.cubes_dir)
        sys.exit(1)

    try:
        build_dataset(
            gpkg_path=args.gpkg,
            cubes_dir=args.cubes_dir,
            output_path=args.output,
            bands=args.bands,
            indices=args.indices,
            days_window=args.days_window,
        )
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
