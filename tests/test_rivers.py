import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry.polygon import Polygon

from eoflow.rivers import get_river_network_from_shape

EXETER_POLY = Polygon((
    (-3.5339, 50.7284),
    (-3.5252, 50.7234),
    (-3.5252, 50.7134),
    (-3.5339, 50.7084),
    (-3.5425, 50.7134),
    (-3.5425, 50.7234),
    (-3.5339, 50.7284)
    ))

class TestRivers:

    @classmethod
    def setup_class(cls):
        # Example shape in lat/long50.7184, -3.5339
        cls.shp = EXETER_POLY
        cls.rivers, cls.graph = get_river_network_from_shape(cls.shp, return_graph=True)

    def test_gdf_from_shape(self):
        assert isinstance(self.rivers, gpd.GeoDataFrame)
        assert self.rivers.shape[0] > 0
        for col in ['osm_id', 'osm_type', 'geometry', 'tags']:
            assert col in self.rivers.columns

    def test_graph_from_shape(self):
        assert isinstance(self.graph, nx.Graph)
        assert self.graph.edges


if __name__ == '__main__':
    pytest.main()
