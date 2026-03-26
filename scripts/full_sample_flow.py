import webbrowser
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
    # 2.1 Select single row from df and convert to Sample
    SAMPLE_IND = 0
    row_series = wq_sample_df.iloc[SAMPLE_IND]

    cs = Sample(
        row=row_series,
        catchment=None,
    )
    logger.info(f"Created Sample from row {SAMPLE_IND}")

    # Setup checkpointing tools
    checkpoint_path = DATA_DIR / "temp/sample_flow_checkpoint_{:03d}"

    def checkpoint_step(s: int) -> Path:
        return Path(str(checkpoint_path).format(s))

    # 3. Catchment Delineation
    # 3.1 Delineate catchment from DEM
    chk = checkpoint_step(1)
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
    chk = checkpoint_step(2)
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

    # 5. EO Queries (NDVI + NDWI)
    # Fetch NDVI and NDWI for the catchment using openEO / Sentinel-2.
    # A separate checkpoint is used so the potentially slow download is not
    # repeated on re-runs.
    EO_START_DATE = "2024-02-01"
    EO_END_DATE = "2024-04-30"
    chk = checkpoint_step(3)
    if chk.exists() and chk.is_dir():
        cs = Sample.load(chk)
        logger.info(f"Loaded EO data from checkpoint {chk}")
    else:
        try:
            logger.info(
                f"Connecting to openEO to fetch NDVI and NDWI ({EO_START_DATE} → {EO_END_DATE}) …"
            )
            conn = Sample.connect_openeo()

            cs.fetch_ndvi(conn, start_date=EO_START_DATE, end_date=EO_END_DATE)
            logger.info(f"NDVI fetched for sample {cs.id}")

            cs.fetch_ndwi(conn, start_date=EO_START_DATE, end_date=EO_END_DATE)
            logger.info(f"NDWI fetched for sample {cs.id}")

            cs.save(chk)
            logger.info(f"Saved EO data to checkpoint {chk}")
        except Exception as exc:
            logger.warning(f"EO query step skipped — could not fetch NDVI/NDWI: {exc}")

    # 6. Visualize
    # 6.1 Load county boundary
    COUNTY_BOUNDARY_DIR = DATA_DIR / "shapefiles/devon_county"
    assert COUNTY_BOUNDARY_DIR.exists() and COUNTY_BOUNDARY_DIR.is_dir()
    county_boundary = gpd.read_file(COUNTY_BOUNDARY_DIR.glob("*.shp").__next__())

    # 6.2 Generate base folium map with county boundary
    m = make_base_map_from_poly(county_boundary)

    # 6.3 Add county boundary
    add_area(county_boundary, name="County Boundary", f_map=m)

    # 6.4 Add sample catchment boundary
    add_sample(cs, f_map=m, catchment_color="green", raster_layers="auto")

    # 6.5 Finalise the layer control once all layers are present, then render.
    folium.LayerControl(collapsed=False).add_to(m)

    # 6.6 Save visualisation to HTML and open in browser (non-blocking)
    output_path = PROJECT_ROOT / "outputs/sample_flow_map.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info(f"Map saved to {output_path}")
    webbrowser.open(output_path.as_uri())

    logger.info(f"script {__file__} finished without error!")


if __name__ == "__main__":
    main()
