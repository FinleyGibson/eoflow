"""
eoflow/features.py
───────────────────
Catchment-scale feature extraction for regression modelling.

Organises predictors into two tiers:

  Direct metrics
  ──────────────
  • Identity / target  — notation, site name, result (turbidity), date
  • Catchment geometry — area, perimeter, compactness ratio
  • Terrain            — elevation, slope, aspect (BNG 50 m rasters)
  • Vegetation (NDVI)  — mean, spread, bare-earth & dense-cover fractions,
                         temporal amplitude (UTM 10 m raster)
  • Moisture (NDWI)    — mean, wet-area fraction (UTM 10 m raster)
  • Soil (SEARG)       — fractional coverage per group, area-weighted BFI &
                         SPR, Shannon entropy, aggregate erodible fraction
  • Rainfall           — mean rate, peak rate, total depth, spatial CV

  Interaction metrics  (cross-layer joint / product features)
  ───────────────────
  • Bare-earth × steep-slope fraction
  • Erodible-soil × bare-earth fraction
  • Wet-area × steep-slope fraction
  • Rainfall-mean × bare-earth fraction

Public API
──────────
  extract_features(sample, **kwargs) -> pd.Series
      Extract all features from a single :class:`~eoflow.samples.Sample`.

  extract_features_batch(samples_dir, **kwargs) -> pd.DataFrame
      Walk a directory of saved Sample subdirectories and return a DataFrame
      with one row per sample.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform

from eoflow.log_utils import get_logger
from eoflow.samples import Sample
from eoflow.soil import SEARG_GROUPS

logger = get_logger("eoflow.scripts.features")

# ---------------------------------------------------------------------------
# Tuneable thresholds (overridable via keyword args to extract_features)
# ---------------------------------------------------------------------------

#: NDVI below this value → "bare / sparse cover"
BARE_NDVI_THRESHOLD: float = 0.2

#: NDVI above this value → "dense vegetation"
DENSE_NDVI_THRESHOLD: float = 0.6

#: NDWI above this value → "open water / saturated surface"
WET_NDWI_THRESHOLD: float = 0.0

#: Primary steep-slope threshold (°)
STEEP_SLOPE_LOW: float = 10.0

#: Secondary steep-slope threshold (°)
STEEP_SLOPE_HIGH: float = 15.0

#: SEARG groups whose soils are considered structurally erodible
#: (light-textured, shallow, or alluvial — low cohesion / high detachability)
ERODIBLE_GROUPS: frozenset[str] = frozenset(
    {
        "Alluvial and coastal soils",
        "Light free drainage",
        "Light soils with moderate & poor drainage",
        "Shallow soils",
    }
)

# CRS for Sentinel-2 / openEO layers (UTM zone 30N — covers Devon)
_EO_CRS: str = "EPSG:32630"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_col(text: str) -> str:
    """Convert a free-text label to a safe DataFrame column fragment."""
    return (
        text.lower()
        .replace(" & ", "_and_")
        .replace("&", "_and_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .strip("_")
    )


def _eo_catchment_mask(
    catchment_geom,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """Boolean mask (True = inside catchment) for an EO raster in ``_EO_CRS``.

    Parameters
    ----------
    catchment_geom:
        Catchment polygon in WGS-84 (EPSG:4326).
    x_coords, y_coords:
        1-D pixel-centre coordinate arrays for the EO grid.
        x is ascending; y is typically descending (north-up convention).

    Returns
    -------
    inside : np.ndarray, shape (len(y_coords), len(x_coords)), dtype bool
    """
    catchment_utm = (
        gpd.GeoSeries([catchment_geom], crs="EPSG:4326")
        .to_crs(_EO_CRS)
        .iloc[0]
    )
    x_c = np.asarray(x_coords, dtype=float)
    y_c = np.asarray(y_coords, dtype=float)
    dx = x_c[1] - x_c[0]
    dy = y_c[1] - y_c[0]  # negative for north-up rasters

    # from_origin expects (west, north, cell_width, cell_height) with positive sizes
    affine = rasterio.transform.from_origin(
        x_c[0] - abs(dx) / 2.0,
        y_c[0] + abs(dy) / 2.0,  # north edge of the first (topmost) row
        abs(dx),
        abs(dy),
    )
    outside = rasterio.features.geometry_mask(
        [catchment_utm.__geo_interface__],
        out_shape=(len(y_c), len(x_c)),
        transform=affine,
        invert=False,
    )
    return ~outside  # True = inside


def _eo_prepare(da) -> np.ndarray:
    """Coerce an EO DataArray to a float32 ndarray, handling object dtype."""
    vals = da.values
    if np.issubdtype(vals.dtype, np.floating):
        return vals.astype(np.float32)

    if vals.dtype.kind == "O":
        def _to_f(v) -> float:
            if v in (b"", b"nan", "", None):
                return np.nan
            try:
                return float(np.float32(v))  # type: ignore[arg-type]
            except (ValueError, TypeError):
                return np.nan
        return np.vectorize(_to_f)(vals).astype(np.float32)

    return vals.astype(np.float32)


def _eo_time_mean(
    da,
    inside: np.ndarray,
    *,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """Time-mean of an EO DataArray, clipped to [vmin, vmax] and masked to catchment.

    Returns a 2-D float array (y × x); pixels outside the catchment or
    outside the valid range are NaN.
    """
    vals = _eo_prepare(da).astype(float)  # (t, y, x)

    # Clip to physically valid index range
    vals = np.where((vals >= vmin) & (vals <= vmax), vals, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_2d = np.nanmean(vals, axis=0)  # (y, x)

    mean_2d[~inside] = np.nan
    return mean_2d


# ---------------------------------------------------------------------------
# Feature group extractors
# ---------------------------------------------------------------------------


def _identity_features(sample: Sample) -> dict:
    """Sample identity and regression target."""
    log_flow = (
        math.log1p(sample.flow_acc_at_pour_point)
        if sample.flow_acc_at_pour_point is not None
        else np.nan
    )
    return {
        "notation": sample.notation,
        "site_name": sample.site_name,
        "date": sample.date,
        "result": sample.result,
        "unit": sample.unit,
        "flow_acc_at_pour_point_log": log_flow,
    }


def _geometry_features(sample: Sample) -> dict:
    """Catchment area, perimeter, and compactness ratio.

    Compactness (Polsby–Popper): 4π·A / P²
    → 1.0 for a perfect circle; → 0 for very elongated shapes.
    """
    nan_row: dict = {
        "catchment_area_km2": np.nan,
        "catchment_perimeter_km": np.nan,
        "catchment_compactness": np.nan,
    }
    if not sample.has_catchment:
        return nan_row

    area_km2 = sample.catchment_area_km2()
    if area_km2 is None:
        return nan_row

    gs = gpd.GeoSeries([sample.catchment], crs="EPSG:4326").to_crs("EPSG:27700")
    perimeter_km = float(gs.length.iloc[0]) / 1_000.0
    compactness = (
        4.0 * math.pi * area_km2 / perimeter_km**2 if perimeter_km > 0 else np.nan
    )

    return {
        "catchment_area_km2": area_km2,
        "catchment_perimeter_km": perimeter_km,
        "catchment_compactness": compactness,
    }


def _terrain_features(
    sample: Sample,
    *,
    steep_low: float = STEEP_SLOPE_LOW,
    steep_high: float = STEEP_SLOPE_HIGH,
) -> dict:
    """Elevation, slope, and aspect statistics from the BNG rasters."""
    out: dict = {
        "elevation_mean_m": np.nan,
        "elevation_range_m": np.nan,
        "elevation_std_m": np.nan,
        "slope_mean_deg": np.nan,
        "slope_std_deg": np.nan,
        "slope_p90_deg": np.nan,
        f"slope_frac_gt{int(steep_low)}deg": np.nan,
        f"slope_frac_gt{int(steep_high)}deg": np.nan,
        "aspect_northness": np.nan,
        "aspect_eastness": np.nan,
    }
    sl = sample.layers

    if sl.topography is not None:
        v = sl.topography.values.astype(float)
        f = v[np.isfinite(v)]
        if f.size:
            out["elevation_mean_m"] = float(f.mean())
            out["elevation_range_m"] = float(f.max() - f.min())
            out["elevation_std_m"] = float(f.std())

    if sl.slope is not None:
        v = sl.slope.values.astype(float)
        f = v[np.isfinite(v)]
        if f.size:
            out["slope_mean_deg"] = float(f.mean())
            out["slope_std_deg"] = float(f.std())
            out["slope_p90_deg"] = float(np.percentile(f, 90))
            out[f"slope_frac_gt{int(steep_low)}deg"] = float((f > steep_low).mean())
            out[f"slope_frac_gt{int(steep_high)}deg"] = float((f > steep_high).mean())

    if sl.aspect is not None:
        v = sl.aspect.values.astype(float)
        f = v[np.isfinite(v)]
        if f.size:
            rad = np.deg2rad(f)
            out["aspect_northness"] = float(np.cos(rad).mean())
            out["aspect_eastness"] = float(np.sin(rad).mean())

    return out


def _ndvi_features(
    sample: Sample,
    *,
    bare_threshold: float = BARE_NDVI_THRESHOLD,
    dense_threshold: float = DENSE_NDVI_THRESHOLD,
) -> dict:
    """NDVI-derived vegetation cover statistics."""
    out: dict = {
        "ndvi_mean": np.nan,
        "ndvi_std": np.nan,
        "ndvi_p10": np.nan,
        "ndvi_temporal_amplitude": np.nan,
        "ndvi_bare_frac": np.nan,
        "ndvi_dense_frac": np.nan,
    }
    sl = sample.layers
    if sl.ndvi is None or not sample.has_catchment:
        return out

    ndvi = sl.ndvi
    inside = _eo_catchment_mask(
        sample.catchment,
        ndvi.coords["x"].values,
        ndvi.coords["y"].values,
    )
    mean_2d = _eo_time_mean(ndvi, inside)
    finite = mean_2d[np.isfinite(mean_2d)]

    if not finite.size:
        logger.warning("No finite NDVI pixels for %s — skipping NDVI features.", sample.notation)
        return out

    out["ndvi_mean"] = float(finite.mean())
    out["ndvi_std"] = float(finite.std())
    out["ndvi_p10"] = float(np.percentile(finite, 10))
    out["ndvi_bare_frac"] = float((finite < bare_threshold).mean())
    out["ndvi_dense_frac"] = float((finite > dense_threshold).mean())

    # Temporal amplitude: per-pixel (max − min) over time → spatial mean
    vals = _eo_prepare(ndvi).astype(float)
    vals = np.where((vals >= -1.0) & (vals <= 1.0), vals, np.nan)
    vals[:, ~inside] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        px_amp = np.nanmax(vals, axis=0) - np.nanmin(vals, axis=0)
    amp_f = px_amp[np.isfinite(px_amp)]
    if amp_f.size:
        out["ndvi_temporal_amplitude"] = float(amp_f.mean())

    return out


def _ndwi_features(
    sample: Sample,
    *,
    wet_threshold: float = WET_NDWI_THRESHOLD,
) -> dict:
    """NDWI-derived surface moisture statistics."""
    out: dict = {
        "ndwi_mean": np.nan,
        "ndwi_wet_frac": np.nan,
    }
    sl = sample.layers
    if sl.ndwi is None or not sample.has_catchment:
        return out

    ndwi = sl.ndwi
    inside = _eo_catchment_mask(
        sample.catchment,
        ndwi.coords["x"].values,
        ndwi.coords["y"].values,
    )
    mean_2d = _eo_time_mean(ndwi, inside)
    finite = mean_2d[np.isfinite(mean_2d)]

    if not finite.size:
        logger.warning("No finite NDWI pixels for %s — skipping NDWI features.", sample.notation)
        return out

    out["ndwi_mean"] = float(finite.mean())
    out["ndwi_wet_frac"] = float((finite > wet_threshold).mean())
    return out


def _soil_features(sample: Sample) -> dict:
    """SEARG fractional coverage, BFI/SPR, Shannon entropy, erodible fraction."""
    # Initialise with NaN for all 12 groups + summary stats
    out: dict = {
        f"soil_frac_{_safe_col(g)}": np.nan for g in SEARG_GROUPS
    }
    out.update(
        {
            "soil_bfi_weighted_mean": np.nan,
            "soil_spr_weighted_mean": np.nan,
            "soil_entropy": np.nan,
            "soil_erodible_frac": np.nan,
        }
    )

    sl = sample.layers
    if sl.soil_type is None:
        return out

    # Per-group fractions
    for g in SEARG_GROUPS:
        out[f"soil_frac_{_safe_col(g)}"] = float(sl.soil_type.get(g, 0.0))

    # Shannon entropy H = −Σ p·ln(p)  (higher → more diverse soil mosaic)
    fracs = np.array([sl.soil_type.get(g, 0.0) for g in SEARG_GROUPS], dtype=float)
    pos = fracs[fracs > 0]
    if pos.size:
        out["soil_entropy"] = float(-np.sum(pos * np.log(pos)))

    # Erodible-soil fraction (sum of structurally erodible groups)
    out["soil_erodible_frac"] = float(
        sum(sl.soil_type.get(g, 0.0) for g in ERODIBLE_GROUPS)
    )

    # Area-weighted BFI and SPR from raw polygon geometries
    if (
        sl.soil_polygons is not None
        and not sl.soil_polygons.empty
        and sample.has_catchment
    ):
        try:
            catchment_bng = (
                gpd.GeoSeries([sample.catchment], crs="EPSG:4326")
                .to_crs("EPSG:27700")
                .iloc[0]
            )
            sp = sl.soil_polygons.copy()
            sp["_clip_geom"] = sp.geometry.intersection(catchment_bng)
            sp = sp[sp["_clip_geom"].notna() & ~sp["_clip_geom"].is_empty].copy()
            sp["_clip_area"] = sp["_clip_geom"].area
            total_area = sp["_clip_area"].sum()
            if total_area > 0:
                bfi = sp["BFI"].astype(float)
                spr = sp["SPR"].astype(float)
                out["soil_bfi_weighted_mean"] = float(
                    (bfi * sp["_clip_area"]).sum() / total_area
                )
                out["soil_spr_weighted_mean"] = float(
                    (spr * sp["_clip_area"]).sum() / total_area
                )
        except Exception as exc:
            logger.warning(
                "BFI/SPR computation failed for %s: %s", sample.notation, exc
            )

    return out


def _rainfall_features(sample: Sample) -> dict:
    """Rainfall statistics from the Met Office UKV layer."""
    out: dict = {
        "rainfall_mean_mm_h": np.nan,
        "rainfall_max_mm_h": np.nan,
        "rainfall_total_depth_mm": np.nan,
        "rainfall_spatial_cv": np.nan,
    }
    sl = sample.layers
    if sl.rainfall is None:
        return out

    rain = sl.rainfall.values.astype(float)
    finite = rain[np.isfinite(rain)]
    if not finite.size:
        return out

    out["rainfall_mean_mm_h"] = float(finite.mean())
    out["rainfall_max_mm_h"] = float(finite.max())

    # Total depth: sum over all timesteps × timestep length in hours
    try:
        times = sl.rainfall.coords["time"].values
        if len(times) > 1:
            dt_h = float(
                (times[1] - times[0]) / np.timedelta64(1, "h")
            )
            spatial_totals = np.nansum(rain, axis=0)  # sum over time per pixel
            out["rainfall_total_depth_mm"] = float(
                np.nanmean(spatial_totals) * dt_h
            )
    except Exception as exc:
        logger.debug("Rainfall total depth computation failed: %s", exc)

    # Spatial CV: std of catchment-mean timeseries / mean
    if rain.ndim == 3:
        ts = np.nanmean(rain, axis=(1, 2))  # one value per timestep
        mu = float(np.nanmean(ts))
        if mu > 0:
            out["rainfall_spatial_cv"] = float(np.nanstd(ts) / mu)

    return out


def _interaction_features(direct: dict) -> dict:
    """Cross-layer product features derived from the already-computed scalar metrics.

    All features are products or joint fractions of two direct metrics.
    NaN propagates naturally through multiplication.
    """
    bare_frac = direct.get("ndvi_bare_frac", np.nan)
    wet_frac = direct.get("ndwi_wet_frac", np.nan)
    erodible_frac = direct.get("soil_erodible_frac", np.nan)
    steep_10 = direct.get("slope_frac_gt10deg", np.nan)
    steep_15 = direct.get("slope_frac_gt15deg", np.nan)
    rain_mean = direct.get("rainfall_mean_mm_h", np.nan)
    spr = direct.get("soil_spr_weighted_mean", np.nan)
    ndvi_mean = direct.get("ndvi_mean", np.nan)

    return {
        # Bare earth on steep ground: high erosion risk
        "interaction_bare_x_steep10": bare_frac * steep_10,
        "interaction_bare_x_steep15": bare_frac * steep_15,
        # Erodible soil that is also bare: direct sediment supply
        "interaction_erodible_x_bare": erodible_frac * bare_frac,
        # Saturated area on steep ground: saturation-excess runoff
        "interaction_wet_x_steep10": wet_frac * steep_10,
        # Rainfall intensity on bare / unprotected surfaces
        "interaction_rain_x_bare": rain_mean * bare_frac,
        # High-SPR soil with low vegetation cover
        "interaction_spr_x_bare": spr * bare_frac,
        # Inverse vegetation × slope: steeper slopes with less cover
        "interaction_slope_mean_x_inv_ndvi": (
            direct.get("slope_mean_deg", np.nan) * (1.0 - ndvi_mean)
            if not np.isnan(ndvi_mean) else np.nan
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(
    sample: Sample,
    *,
    bare_ndvi_threshold: float = BARE_NDVI_THRESHOLD,
    dense_ndvi_threshold: float = DENSE_NDVI_THRESHOLD,
    wet_ndwi_threshold: float = WET_NDWI_THRESHOLD,
    steep_slope_low: float = STEEP_SLOPE_LOW,
    steep_slope_high: float = STEEP_SLOPE_HIGH,
) -> pd.Series:
    """Extract all catchment-scale features from a single :class:`~eoflow.samples.Sample`.

    Parameters
    ----------
    sample:
        A loaded :class:`~eoflow.samples.Sample` instance.
    bare_ndvi_threshold:
        NDVI below which a pixel is classified as bare / sparse cover
        (default 0.2).
    dense_ndvi_threshold:
        NDVI above which a pixel is classified as dense vegetation
        (default 0.6).
    wet_ndwi_threshold:
        NDWI above which a pixel is classified as open water / saturated
        (default 0.0).
    steep_slope_low:
        Primary steep-slope threshold in degrees (default 10°).
    steep_slope_high:
        Secondary steep-slope threshold in degrees (default 15°).

    Returns
    -------
    pd.Series
        One entry per feature; NaN where the required layer is absent.
        Feature groups are ordered: identity → geometry → terrain →
        NDVI → NDWI → soil → rainfall → interactions.
    """
    logger.info("Extracting features for %s …", sample.notation)

    direct: dict = {}
    direct.update(_identity_features(sample))
    direct.update(_geometry_features(sample))
    direct.update(
        _terrain_features(sample, steep_low=steep_slope_low, steep_high=steep_slope_high)
    )
    direct.update(
        _ndvi_features(
            sample,
            bare_threshold=bare_ndvi_threshold,
            dense_threshold=dense_ndvi_threshold,
        )
    )
    direct.update(_ndwi_features(sample, wet_threshold=wet_ndwi_threshold))
    direct.update(_soil_features(sample))
    direct.update(_rainfall_features(sample))

    # Interaction features reference the scalar direct metrics computed above
    direct.update(_interaction_features(direct))

    logger.info(
        "  %d features extracted (%d NaN) for %s",
        len(direct),
        sum(1 for v in direct.values() if isinstance(v, float) and math.isnan(v)),
        sample.notation,
    )
    return pd.Series(direct)


def extract_features_batch(
    samples_dir: Path | str,
    **kwargs,
) -> pd.DataFrame:
    """Extract features for every saved Sample found under *samples_dir*.

    Each immediate subdirectory of *samples_dir* that contains a
    ``metadata.json`` file is treated as a saved Sample and loaded with
    :meth:`~eoflow.samples.Sample.load`.

    Parameters
    ----------
    samples_dir:
        Root directory to scan (e.g. ``data/sample_instances``).
    **kwargs:
        Forwarded to :func:`extract_features`.

    Returns
    -------
    pd.DataFrame
        One row per sample, columns ordered as per :func:`extract_features`.
        Samples that fail to load or extract are skipped with a warning.
    """
    samples_dir = Path(samples_dir)
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    # Collect all subdirectories that look like saved Samples
    candidate_dirs = sorted(
        p for p in samples_dir.iterdir()
        if p.is_dir() and (p / "metadata.json").exists()
    )
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No valid Sample directories found in {samples_dir}. "
            "Each subdirectory must contain a metadata.json file."
        )

    logger.info(
        "Found %d Sample director%s under %s",
        len(candidate_dirs),
        "y" if len(candidate_dirs) == 1 else "ies",
        samples_dir,
    )

    rows: list[pd.Series] = []
    for sample_dir in candidate_dirs:
        try:
            sample = Sample.load(sample_dir)
            row = extract_features(sample, **kwargs)
            rows.append(row)
            logger.info("  ✓ %s", sample_dir.name)
        except Exception as exc:
            logger.warning("  ✗ %s — skipped: %s", sample_dir.name, exc)

    if not rows:
        raise RuntimeError(
            f"Feature extraction failed for all {len(candidate_dirs)} sample(s)."
        )

    df = pd.DataFrame(rows).reset_index(drop=True)
    logger.info("Batch extraction complete: %d rows × %d columns.", *df.shape)
    return df
