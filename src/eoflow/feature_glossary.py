"""
eoflow/feature_glossary.py
─────────────────────────────
Human-readable descriptions for the catchment-scale predictors produced by
:func:`eoflow.features.extract_features`. Used by the model-analysis
notebooks (`notebooks/model_analysis/`) to render a "what does this feature
mean" table for whichever features a given model actually surfaced as
important — so the table reflects the model being described instead of a
fixed, hand-picked list.

Public API
──────────
  FEATURE_DESCRIPTIONS
      dict[str, str] — one entry per predictor column emitted by
      :func:`eoflow.features.extract_features`.

  describe_features(feature_names) -> pd.DataFrame
      Build a Feature / Group / Description table for a given list of
      feature names, in that order.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

#: Description for every predictor column extract_features() can emit.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "flow_acc_at_pour_point_log": (
        "Log1p of D8 flow accumulation at the catchment pour point — a proxy for "
        "stream size / position in the drainage network."
    ),
    "catchment_area_km2": "Delineated catchment area, in km².",
    "catchment_perimeter_km": "Delineated catchment boundary length, in km.",
    "catchment_compactness": (
        "Polsby–Popper compactness ratio, 4πA/P² — 1.0 for a circular catchment, "
        "→ 0 for elongated shapes."
    ),
    "elevation_mean_m": "Mean elevation across the catchment (DEM), in metres.",
    "elevation_range_m": "Elevation range (max − min) across the catchment, in metres.",
    "elevation_std_m": "Standard deviation of elevation across the catchment, in metres.",
    "slope_mean_deg": "Mean terrain slope across the catchment, in degrees.",
    "slope_std_deg": "Standard deviation of terrain slope across the catchment, in degrees.",
    "slope_p90_deg": "90th-percentile terrain slope across the catchment, in degrees.",
    "slope_frac_gt10deg": "Fraction of catchment area with slope steeper than 10°.",
    "slope_frac_gt15deg": "Fraction of catchment area with slope steeper than 15°.",
    "aspect_northness": (
        "Mean cos(aspect) across the catchment — +1 = due north-facing, "
        "-1 = due south-facing."
    ),
    "aspect_eastness": (
        "Mean sin(aspect) across the catchment — +1 = due east-facing, "
        "-1 = due west-facing."
    ),
    "ndvi_mean": (
        "Mean NDVI (Normalized Difference Vegetation Index) across the catchment — "
        "overall greenness / vegetation density (Sentinel-2)."
    ),
    "ndvi_std": "Standard deviation of NDVI across catchment pixels — vegetation-cover heterogeneity.",
    "ndvi_p10": "10th-percentile NDVI across the catchment — captures the sparsest-vegetation tail.",
    "ndvi_temporal_amplitude": (
        "Mean per-pixel (max − min) NDVI over the observation window — seasonal "
        "vegetation variability."
    ),
    "ndvi_bare_frac": (
        "Fraction of catchment pixels classified as bare/sparse cover "
        "(NDVI below the bare-earth threshold)."
    ),
    "ndvi_dense_frac": (
        "Fraction of catchment pixels classified as dense vegetation "
        "(NDVI above the dense-cover threshold)."
    ),
    "ndwi_mean": (
        "Mean NDWI (Normalized Difference Water Index) across the catchment — overall "
        "surface moisture / water presence (Sentinel-2)."
    ),
    "ndwi_wet_frac": (
        "Fraction of catchment pixels with NDWI above the wet threshold — open water / "
        "saturated surface extent."
    ),
    "soil_frac_alluvial_and_coastal_soils": (
        'Fractional coverage of the SEARG "Alluvial and coastal soils" group.'
    ),
    "soil_frac_heavy_clay_soils_with_poor_drainage": (
        'Fractional coverage of the SEARG "Heavy clay soils with poor drainage" group.'
    ),
    "soil_frac_light_free_drainage": 'Fractional coverage of the SEARG "Light free drainage" group.',
    "soil_frac_light_soils_with_moderate_and_poor_drainage": (
        'Fractional coverage of the SEARG "Light soils with moderate & poor drainage" group.'
    ),
    "soil_frac_man_made": 'Fractional coverage of the SEARG "Man made" group.',
    "soil_frac_medium_soils_with_free_drainage": (
        'Fractional coverage of the SEARG "Medium soils with free drainage" group.'
    ),
    "soil_frac_medium_soils_with_moderate_drainage": (
        'Fractional coverage of the SEARG "Medium soils with moderate drainage" group.'
    ),
    "soil_frac_medium_with_soils_poor_drainage": (
        'Fractional coverage of the SEARG "Medium with soils poor drainage" group.'
    ),
    "soil_frac_organic_soils_with_free_drainage": (
        'Fractional coverage of the SEARG "Organic soils with free drainage" group.'
    ),
    "soil_frac_organic_soils_with_poor_drainage": (
        'Fractional coverage of the SEARG "Organic soils with poor drainage" group.'
    ),
    "soil_frac_peat": 'Fractional coverage of the SEARG "Peat" soils group.',
    "soil_frac_shallow_soils": 'Fractional coverage of the SEARG "Shallow soils" group.',
    "soil_bfi_weighted_mean": (
        "Area-weighted mean Baseflow Index — higher values indicate more permeable, "
        "slower-draining soils (more baseflow, less quick runoff)."
    ),
    "soil_spr_weighted_mean": (
        "Area-weighted mean Standard Percentage Runoff — higher values indicate soils "
        "that generate more direct runoff."
    ),
    "soil_entropy": (
        "Shannon entropy of the SEARG group-fraction distribution — higher values "
        "indicate a more diverse soil mosaic."
    ),
    "soil_erodible_frac": (
        "Aggregate fraction of structurally erodible soil groups (alluvial/coastal, "
        "light free-drainage, light soils with moderate & poor drainage, shallow soils)."
    ),
    "rainfall_mean_mm_h": (
        "Mean NIMROD radar rainfall rate (mm/h) over the catchment during the lookback window."
    ),
    "rainfall_max_mm_h": (
        "Peak NIMROD radar rainfall rate (mm/h) over the catchment during the lookback window."
    ),
    "rainfall_total_depth_mm": (
        "Total rainfall depth (mm) accumulated over the catchment during the lookback window."
    ),
    "rainfall_spatial_cv": (
        "Coefficient of variation of the catchment-mean rainfall timeseries — rainfall "
        "variability over the lookback window."
    ),
    "interaction_bare_x_steep10": (
        "ndvi_bare_frac × slope_frac_gt10deg — bare ground on moderately steep (>10°) terrain."
    ),
    "interaction_bare_x_steep15": (
        "ndvi_bare_frac × slope_frac_gt15deg — joint exposure of steep (>15°), "
        "sparsely-vegetated terrain (high erosion-risk zones)."
    ),
    "interaction_erodible_x_bare": (
        "soil_erodible_frac × ndvi_bare_frac — erodible soil that is also bare; a direct "
        "sediment-supply proxy."
    ),
    "interaction_wet_x_steep10": (
        "ndwi_wet_frac × slope_frac_gt10deg — saturated ground on steep terrain "
        "(saturation-excess runoff risk)."
    ),
    "interaction_rain_x_bare": (
        "rainfall_mean_mm_h × ndvi_bare_frac — rainfall intensity landing on "
        "bare/sparsely-vegetated ground; a proxy for erosive rainfall on unprotected soil."
    ),
    "interaction_spr_x_bare": (
        "soil_spr_weighted_mean × ndvi_bare_frac — high-runoff-potential soil that is also bare."
    ),
    "interaction_slope_mean_x_inv_ndvi": (
        "slope_mean_deg × (1 − ndvi_mean) — mean steepness weighted by lack of vegetation cover."
    ),
}

#: (prefix, group label) pairs, checked in order, for grouping a feature name.
_GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("catchment_", "Geometry"),
    ("elevation_", "Terrain"),
    ("slope_", "Terrain"),
    ("aspect_", "Terrain"),
    ("ndvi_", "NDVI"),
    ("ndwi_", "NDWI"),
    ("soil_", "Soil"),
    ("rainfall_", "Rainfall"),
    ("interaction_", "Interaction"),
)


def _feature_group(name: str) -> str:
    """Best-guess feature group for *name*, by column-name prefix."""
    for prefix, group in _GROUP_PREFIXES:
        if name.startswith(prefix):
            return group
    if name == "flow_acc_at_pour_point_log":
        return "Terrain"
    return "Other"


def describe_features(feature_names: Iterable[str]) -> pd.DataFrame:
    """Build a Feature / Group / Description table for *feature_names*.

    Parameters
    ----------
    feature_names:
        Feature column names (e.g. a fitted model's `coefficients.index`),
        in the order they should appear in the output.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``group``, ``description``. Rows follow the
        order of *feature_names*. Names without a known description fall
        back to a placeholder rather than raising, since notebooks call this
        with whatever features a given model happened to select.
    """
    return pd.DataFrame(
        {
            "feature": name,
            "group": _feature_group(name),
            "description": FEATURE_DESCRIPTIONS.get(name, "No description available."),
        }
        for name in feature_names
    )
