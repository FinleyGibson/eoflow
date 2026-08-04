"""
Scripts for eoflow data acquisition, processing, and visualisation.
"""

__all__ = [
    # EA water quality data
    "get_ea_water_quality",
    "ge_ea_water_quality_by_shapefile",
    "convert_ea_csv",
    # Earth observation (openEO) data
    "get_eo_data_by_shapefile",
    # DEM
    "get_dem",
    # Catchment delineation
    "delineate_catchments",
    "delineate_catchment",
    # NIMROD rainfall
    "process_nimrod_local",
    # Sample preparation and feature extraction
    "prepare_samples",
    "extract_features",
    # Visualisation
    "visualise_dem",
    "visualise_ea_samples",
    "visualise_catchments",
    "visualise_nimrod",
    "full_sample_flow",
    # Utilities
    "run_api",
    "test_nimrod_rainfall",
]
