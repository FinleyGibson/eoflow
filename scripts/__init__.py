"""
Scripts for eoflow data acquisition, processing, and visualisation.
"""

__all__ = [
    # EA water quality data
    "get_ea_water_quality",
    "get_ea_water_quality_by_shapefile",
    "convert_ea_csv",
    # DEM
    "get_dem",
    "get_devon_dem",
    "get_devon_shapefile",
    # Catchment delineation
    "delineate_catchments",
    "delineate_catchment",
    "downsample_gpkg",
    "gpkg_report",
    # NIMROD rainfall
    "nimrod_process_local",
    # Sample preparation and feature extraction
    "prepare_samples",
    "extract_features",
    "feature_report",
    # Visualisation
    "visualise_dem",
    "visualise_ea_samples",
    "visualise_catchments",
    "visualise_nimrod",
    "full_sample_flow",
    # Utilities
    "run_api",
    "sample_report",
    "test_nimrod_rainfall",
]
