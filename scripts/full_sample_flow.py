from datetime import datetime
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd

from eoflow.folium import add_area, add_sample, make_base_map_from_poly
from eoflow.log_utils import get_logger
from eoflow.samples import Sample
from eoflow.utils import DATA_DIR, PROJECT_ROOT

# 0 Set up logging
logger = get_logger("eoflow.scripts.full_sample_flow")

# 1. Load Required Data


def main():
    # 1.1 Load water quality sample data to Dataframe
    wq_sample_path = PROJECT_ROOT / "data/wq_samples/devon_water_quality_10.csv"
    assert wq_sample_path.exists() and wq_sample_path.suffix == ".csv"
    logger.info(f"Loading water quality sample data from {wq_sample_path}")

    wq_sample_df = pd.read_csv(wq_sample_path)
    assert wq_sample_df.shape[0] > 1
    logger.info(f"Loaded {wq_sample_df.shape[0]} water quality samples")

    # 1.2 Load Devon DEM file
    devon_dem_path = DATA_DIR / "dems/devon_dem_cop30.tif"
    assert devon_dem_path.exists() and devon_dem_path.suffix == ".tif"
    logger.info(f"Loaded Devon DEM from {devon_dem_path}")

    # 2. Create single Sample
    # 2.1 Select single row from df and convert to CatchmentSample
    SAMPLE_IND = 0
    row_series = wq_sample_df.iloc[SAMPLE_IND]

    cs = Sample(
        row=row_series,
        catchment=None,
    )
    logger.info(f"Created CatchmentSample from row {SAMPLE_IND}")

    # Setup checkpointing tools
    checkpoint_path = DATA_DIR / "temp/sample_flow_checkpoint_{}"
    checkpoint_step = lambda s: Path(str(checkpoint_path).format(s))

    # 3. Catchment Delineation
    # 3.1 Delineate catchment from DEM
    chk = checkpoint_step("001")
    if chk.exists() and chk.is_dir():
        # load from checkpoint if it exists
        cs = Sample.load(chk)
        logger.info(f"Loaded catchment from checkpoint {chk}")
    else:
        # otherwise, delineate from DEM and save checkpoint
        FLOW_ACC_THRESH = 5000
        assert cs.catchment is None
        cs.delineate(dem_path=devon_dem_path, flow_acc_threshold=FLOW_ACC_THRESH)
        assert cs.catchment is not None
        logger.info(f"Delineated catchment from DEM for sample {cs.id}")
        cs.save(chk)
        logger.info(f"Saved sample data to checkpoint {chk}")

    # 4. Compute layers
    # 4.1 Compute topography & slope
    START_DATETIME = datetime.strptime("2024-03-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
    END_DATETIME = datetime.strptime("2024-03-01T01:00:00", "%Y-%m-%dT%H:%M:%S")
    RAINFALL_DOWNLOAD_PATH = Path(PROJECT_ROOT / "data/temp/rainfall_001")
    chk = checkpoint_step("002")
    if chk.exists() and chk.is_dir():
        # load from checkpoint if it exists
        cs = Sample.load(chk)
        logger.info(f"Loaded catchment from checkpoint {chk}")
    else:
        cs.compute_layers(
            dem_path=DATA_DIR / "dems/devon_dem_cop30.tif",
            rainfall_start=START_DATETIME,
            rainfall_end=END_DATETIME,
            rainfall_download_dir=RAINFALL_DOWNLOAD_PATH,
        )
        logger.info(f"Computed layers for sample {cs.id}")
        cs.save(chk)
        logger.info(f"Saved sample data to checkpoint {chk}")

    # 4. Visualize
    # 4.1 Load county boundary
    COUNTY_BOUNDARY_DIR = DATA_DIR / "shapefiles/devon_county"
    assert COUNTY_BOUNDARY_DIR.exists() and COUNTY_BOUNDARY_DIR.is_dir()
    county_boundary = gpd.read_file(COUNTY_BOUNDARY_DIR.glob("*.shp").__next__())

    # 4.2 Generate base folium map with county boundary
    m = make_base_map_from_poly(county_boundary)

    # 4.3 Add county boundary
    add_area(county_boundary, name="County Boundary", f_map=m)

    # 4.4 Add sample catchment boundary
    add_sample(cs, f_map=m, catchment_color="green", raster_layers="auto")

    # 4.3 Finalise the layer control once all layers are present, then render.
    folium.LayerControl(collapsed=False).add_to(m)

    # 4.4 write visualisation to html5
    m.show_in_browser()

    logger.info(f"script {__file__} finished without error!")


if __name__ == "__main__":
    main()
