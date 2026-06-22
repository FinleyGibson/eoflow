"""
Scripts for eoflow data acquisition, processing, and visualisation.
"""

__all__ = [
    # EA water quality data
    "get_ea_water_quality",
    "ge_ea_water_quality_by_shapefile",
    "convert_ea_csv",
    # DEM
    "get_dem",
    # Catchment delineation
    "delineate_catchments",
    "delineate_catchment",
    # NIMROD rainfall
    "process_nimrod_local",
    "consolidate_nimrod",
    # Sample preparation and feature extraction
    "prepare_samples",
    "extract_features",
    # Visualisation
    "visualise_dem",
    "visualise_ea_samples",
    "visualise_devon_catchments",
    "visualise_nimrod",
    "full_sample_flow",
    # Utilities
    "convert_rainfall",
    "download_rainfall",
    "run_api",
    "test_nimrod_rainfall",
]
