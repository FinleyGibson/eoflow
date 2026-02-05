# %%
import leafmap
import rasterio

from eoflow.utils import HYDROSHEDS_DATA

# %%
tif_path = HYDROSHEDS_DATA / "eu_msk_3s.tif"
assert tif_path.is_file()

# %%
with rasterio.open(tif_path) as src:
    all_bands = src.read()

print(all_bands.shape)

# %%
m = leafmap.Map(center=[52, 10], zoom=5)


m.add_raster(
    str(tif_path),
    opacity=0.8,
)

m.add_basemap("OpenStreetMap")
m
# %%
m.save("out.html")
