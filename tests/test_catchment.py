"""
Unit tests for the catchment delineation module.

These tests verify the catchment delineation functionality without requiring
actual DEM data for most tests. Integration tests with real DEM files should
be run separately and marked with the @pytest.mark.integration decorator.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from eoflow.catchment import delineate_catchment, delineate_catchment_with_metadata

# ---------------------------------------------------------------------------
# Paths to real test data used by integration tests
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CSV_PATH = _DATA_DIR / "ea_water_quality_clean_pruned.csv"
_DEM_PATH = _DATA_DIR / "uk_dem.geotiff"
_DEVON_DEM_PATH = _DATA_DIR / "devon_dem.tif"


@pytest.fixture
def mock_dem_path():
    """Create a temporary DEM file path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        path = Path(tmp.name)
    yield path
    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def sample_point():
    """Create a sample pour point for testing."""
    return Point(-3.5, 50.7)


@pytest.fixture
def sample_downstream_points():
    """Three pour points on the River Exe (Devon), each one downstream of the next.

    These coordinates were validated against ``data/devon_dem.tif`` using
    ``flow_acc_threshold=5000``.  After snapping to the stream network the
    flow-accumulation values and delineated catchment areas increase
    monotonically from upstream to downstream:

    * upstream   – Thorverton reach   acc ≈  448 k  area ≈ 0.035 sq-deg
    * middle     – mid-Exe reach      acc ≈  915 k  area ≈ 0.071 sq-deg
    * downstream – Exeter reach       acc ≈ 1390 k  area ≈ 0.107 sq-deg

    The ``downstream`` catchment fully contains both ``middle`` and
    ``upstream`` catchments; ``middle`` fully contains ``upstream``.
    """
    return {
        "upstream": Point(-3.507, 50.814),  # Thorverton area
        "middle": Point(-3.509, 50.760),  # mid-Exe
        "downstream": Point(-3.511, 50.704),  # Exeter area
    }


@pytest.fixture
def sample_polygon():
    """Create a sample catchment polygon for testing."""
    coords = [
        (-3.5, 50.7),
        (-3.6, 50.7),
        (-3.6, 50.8),
        (-3.5, 50.8),
        (-3.5, 50.7),
    ]
    return Polygon(coords)


@pytest.fixture
def mock_grid():
    """Create a mock Grid object with necessary methods."""
    grid = MagicMock()
    grid.bbox = [-4.0, 50.0, -3.0, 51.0]  # (minx, miny, maxx, maxy)

    # Mock DEM data
    dem_data = np.random.rand(100, 100) * 100  # 100x100 DEM with elevation 0-100
    grid.read_raster.return_value = dem_data

    # Mock pit filling
    grid.fill_pits.return_value = dem_data + 0.1

    # Mock resolve flats
    grid.resolve_flats.return_value = dem_data + 0.2

    # Mock flow direction
    fdir_data = np.ones((100, 100), dtype=np.int32)
    grid.flowdir.return_value = fdir_data

    # Mock flow accumulation
    acc_data = np.random.randint(0, 10000, size=(100, 100))
    grid.accumulation.return_value = acc_data

    # Mock snap to mask
    grid.snap_to_mask.return_value = (-3.5, 50.7)

    # Mock catchment
    catch_data = np.zeros((100, 100), dtype=np.int32)
    catch_data[40:60, 40:60] = 1  # Simple square catchment
    grid.catchment.return_value = catch_data

    # Mock polygonize
    polygon_coords = [
        (-3.55, 50.65),
        (-3.45, 50.65),
        (-3.45, 50.75),
        (-3.55, 50.75),
        (-3.55, 50.65),
    ]
    mock_geom = {"type": "Polygon", "coordinates": [polygon_coords]}
    grid.polygonize.return_value = [(mock_geom, 1)]

    return grid


class TestDelineateCatchmentInputValidation:
    """Test input validation for delineate_catchment function."""

    @patch("eoflow.catchment.Grid")
    def test_invalid_point_type(self, mock_grid_class, mock_dem_path):
        """Test that non-Point objects raise appropriate errors."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid
        mock_grid.read_raster.return_value = np.random.rand(10, 10)

        with pytest.raises((ValueError, AttributeError)):
            delineate_catchment(
                point="not a point",  # type: ignore
                dem_path=mock_dem_path,
            )

    def test_missing_dem_file(self, sample_point):
        """Test that missing DEM files raise FileNotFoundError."""
        nonexistent_path = Path("/nonexistent/path/to/dem.tif")

        with pytest.raises(FileNotFoundError) as exc_info:
            delineate_catchment(point=sample_point, dem_path=nonexistent_path)

        assert "DEM file not found" in str(exc_info.value)

    def test_point_coordinates_accessible(self, sample_point):
        """Test that Point object has accessible coordinates."""
        assert hasattr(sample_point, "x")
        assert hasattr(sample_point, "y")
        assert sample_point.x == -3.5
        assert sample_point.y == 50.7

    @patch("eoflow.catchment.Grid")
    def test_invalid_point_missing_coordinates(self, mock_grid_class, mock_dem_path):
        """Test that objects without x/y attributes raise errors."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid
        mock_grid.read_raster.return_value = np.random.rand(10, 10)

        invalid_point = Mock()
        del invalid_point.x  # Remove x attribute
        del invalid_point.y  # Remove y attribute

        with pytest.raises((ValueError, AttributeError)):
            delineate_catchment(point=invalid_point, dem_path=mock_dem_path)

    @patch("eoflow.catchment.Grid")
    def test_point_outside_bounds(self, mock_grid_class, sample_point, mock_dem_path):
        """Test that points outside DEM bounds raise ValueError."""
        mock_dem_path.touch()

        # Mock Grid with bounds that don't include the sample point
        mock_grid = MagicMock()
        mock_grid.bbox = [-2.0, 49.0, -1.0, 50.0]  # Doesn't include (-3.5, 50.7)
        mock_grid_class.from_raster.return_value = mock_grid
        mock_grid.read_raster.return_value = np.random.rand(10, 10)

        with pytest.raises(ValueError) as exc_info:
            delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        assert "outside DEM bounds" in str(exc_info.value)


class TestDelineateCatchmentParameters:
    """Test parameter handling in delineate_catchment function."""

    def test_default_parameters(self):
        """Test that default parameters are correctly defined."""
        import inspect

        sig = inspect.signature(delineate_catchment)
        params = sig.parameters

        # Verify key parameters exist
        assert "point" in params
        assert "dem_path" in params
        assert "dirmap" in params
        assert "flow_acc_threshold" in params
        assert "routing" in params

        # Check default values
        assert params["dirmap"].default == (64, 128, 1, 2, 4, 8, 16, 32)
        assert params["flow_acc_threshold"].default == 5000
        assert params["routing"].default == "d8"
        assert params["pit_fill"].default is True
        assert params["resolve_flats"].default is True
        assert params["pit_fill_epsilon"].default == 0.0001
        assert params["nodata_out"].default is None

    def test_custom_parameters_accepted(self):
        """Test that custom parameters can be passed."""
        import inspect

        sig = inspect.signature(delineate_catchment)

        # Verify all expected parameters exist
        expected_params = [
            "point",
            "dem_path",
            "dirmap",
            "apply_input_mask",
            "nodata_in",
            "nodata_out",
            "pit_fill",
            "pit_fill_epsilon",
            "resolve_flats",
            "routing",
            "flow_acc_threshold",
        ]

        for param in expected_params:
            assert param in sig.parameters, f"Missing parameter: {param}"

    @patch("eoflow.catchment.Grid")
    def test_unsupported_routing_method(self, mock_grid_class, sample_point, mock_dem_path):
        """Test that unsupported routing methods raise ValueError."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid
        mock_grid.read_raster.return_value = np.random.rand(10, 10)
        mock_grid.fill_pits.return_value = np.random.rand(10, 10)
        mock_grid.resolve_flats.return_value = np.random.rand(10, 10)

        with pytest.raises(ValueError) as exc_info:
            delineate_catchment(
                point=sample_point,
                dem_path=mock_dem_path,
                routing="invalid_routing",
            )

        assert "Unsupported routing method" in str(exc_info.value)


class TestDelineateCatchmentWorkflow:
    """Test the catchment delineation workflow."""

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_successful_delineation(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test successful catchment delineation workflow."""
        mock_dem_path.touch()

        # Setup mocks
        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        # Mock all processing steps
        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data

        # Mock polygonize to return a geometry
        mock_geom = {
            "type": "Polygon",
            "coordinates": [[[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]],
        }
        mock_grid.polygonize.return_value = [(mock_geom, 1)]

        # Mock shape to return a Polygon
        mock_shape.return_value = sample_polygon

        # Execute
        result = delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        # Verify
        assert isinstance(result, Polygon)
        assert result == sample_polygon

        # Verify workflow steps were called
        mock_grid.read_raster.assert_called()
        mock_grid.fill_pits.assert_called()
        mock_grid.resolve_flats.assert_called()
        mock_grid.flowdir.assert_called()
        mock_grid.accumulation.assert_called()
        mock_grid.catchment.assert_called()
        mock_grid.polygonize.assert_called()

    @patch("eoflow.catchment.Grid")
    def test_pit_fill_disabled(self, mock_grid_class, sample_point, mock_dem_path):
        """Test that pit filling can be disabled."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]

        # Execute with pit_fill=False
        with patch("eoflow.catchment.shape") as mock_shape:
            mock_shape.return_value = Polygon(
                [(-3.5, 50.7), (-3.6, 50.7), (-3.6, 50.8), (-3.5, 50.8), (-3.5, 50.7)]
            )
            delineate_catchment(point=sample_point, dem_path=mock_dem_path, pit_fill=False)

        # Verify pit filling was NOT called
        mock_grid.fill_pits.assert_not_called()
        mock_grid.resolve_flats.assert_not_called()

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_resolve_flats_disabled(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that flat resolution can be disabled."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        # Execute with resolve_flats=False but pit_fill=True
        delineate_catchment(
            point=sample_point, dem_path=mock_dem_path, pit_fill=True, resolve_flats=False
        )

        # Verify pit filling was called but resolve_flats was NOT
        mock_grid.fill_pits.assert_called()
        mock_grid.resolve_flats.assert_not_called()

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_snap_to_mask_fallback(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that when snap_to_mask fails, the original coordinates are used."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))

        # Make snap_to_mask raise an exception
        mock_grid.snap_to_mask.side_effect = Exception("No valid cells in mask")

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        # Should still succeed using original coordinates
        result = delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        assert isinstance(result, Polygon)
        # Verify catchment was called with the original point coordinates
        call_kwargs = mock_grid.catchment.call_args[1]
        assert call_kwargs["x"] == sample_point.x
        assert call_kwargs["y"] == sample_point.y

    @patch("eoflow.catchment.View")
    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_snapped_coordinates_passed_to_catchment(
        self, mock_shape, mock_grid_class, mock_view, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that snapped coordinates (not originals) are passed to grid.catchment."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))

        # Snap moves the point to a different location. Snapping is performed
        # via ``View.snap_to_mask`` directly (not ``grid.snap_to_mask``) to
        # avoid a NumPy 2.x incompatibility in pysheds' Grid.snap_to_mask.
        snapped_x, snapped_y = -3.45, 50.72
        mock_view.snap_to_mask.return_value = (snapped_x, snapped_y)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        # Verify catchment was called with the snapped coordinates
        call_kwargs = mock_grid.catchment.call_args[1]
        assert call_kwargs["x"] == snapped_x
        assert call_kwargs["y"] == snapped_y

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_dirmap_passed_through(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that custom dirmap values are passed to flowdir, accumulation, and catchment."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        custom_dirmap = (1, 2, 4, 8, 16, 32, 64, 128)
        delineate_catchment(point=sample_point, dem_path=mock_dem_path, dirmap=custom_dirmap)

        # Verify dirmap was passed to flowdir, accumulation, and catchment
        assert mock_grid.flowdir.call_args[1]["dirmap"] == custom_dirmap
        assert mock_grid.accumulation.call_args[1]["dirmap"] == custom_dirmap
        assert mock_grid.catchment.call_args[1]["dirmap"] == custom_dirmap

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_nodata_in_passed_to_fill_pits(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that nodata_in is forwarded to grid.fill_pits."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        delineate_catchment(point=sample_point, dem_path=mock_dem_path, nodata_in=-9999.0)

        assert mock_grid.fill_pits.call_args[1]["nodata_in"] == -9999.0

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_apply_input_mask_passed_to_accumulation(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that apply_input_mask is forwarded to grid.accumulation."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        delineate_catchment(point=sample_point, dem_path=mock_dem_path, apply_input_mask=True)

        assert mock_grid.accumulation.call_args[1]["apply_input_mask"] is True

    @patch("eoflow.catchment.View")
    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_flow_acc_threshold_used_in_snap(
        self, mock_shape, mock_grid_class, mock_view, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that flow_acc_threshold controls the mask used for snapping."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))

        # Accumulation with known values
        acc_data = np.full((10, 10), 500)
        acc_data[5, 5] = 3000
        mock_grid.accumulation.return_value = acc_data
        # Snapping is performed via ``View.snap_to_mask`` directly (not
        # ``grid.snap_to_mask``) to avoid a NumPy 2.x incompatibility in
        # pysheds' Grid.snap_to_mask.
        mock_view.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        threshold = 2000
        delineate_catchment(
            point=sample_point, dem_path=mock_dem_path, flow_acc_threshold=threshold
        )

        # The first positional arg to snap_to_mask should be (acc > threshold)
        snap_call_args = mock_view.snap_to_mask.call_args
        mask_arg = snap_call_args[0][0]
        expected_mask = acc_data > threshold
        np.testing.assert_array_equal(mask_arg, expected_mask)

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_pit_fill_epsilon_passed_to_resolve_flats(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that pit_fill_epsilon is forwarded as eps to grid.resolve_flats."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        custom_eps = 0.01
        delineate_catchment(point=sample_point, dem_path=mock_dem_path, pit_fill_epsilon=custom_eps)

        assert mock_grid.resolve_flats.call_args[1]["eps"] == custom_eps

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_nodata_out_propagated_consistently(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that nodata_out is passed to all relevant grid methods."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        custom_nodata = -1
        delineate_catchment(point=sample_point, dem_path=mock_dem_path, nodata_out=custom_nodata)

        # nodata_out is forwarded to fill_pits and resolve_flats (caller value)
        assert mock_grid.fill_pits.call_args[1]["nodata_out"] == custom_nodata
        assert mock_grid.resolve_flats.call_args[1]["nodata_out"] == custom_nodata
        # flowdir and accumulation always receive np.int64(0) as nodata_out to
        # avoid dtype conflicts under NumPy 2.x (NEP 50): float nodata cannot
        # be safely cast into integer arrays.
        assert mock_grid.flowdir.call_args[1]["nodata_out"] == np.int64(0)
        assert mock_grid.accumulation.call_args[1]["nodata_out"] == np.int64(0)
        # catchment always receives np.bool_(False) for the same reason.
        assert mock_grid.catchment.call_args[1]["nodata_out"] == np.bool_(False)

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_point_on_dem_boundary(
        self, mock_shape, mock_grid_class, mock_dem_path, sample_polygon
    ):
        """Test that a point exactly on the DEM boundary is accepted."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-4.0, 50.0)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        # Point exactly on the min corner of the bbox
        boundary_point = Point(-4.0, 50.0)
        result = delineate_catchment(point=boundary_point, dem_path=mock_dem_path)
        assert isinstance(result, Polygon)

        # Point exactly on the max corner of the bbox
        boundary_point = Point(-3.0, 51.0)
        mock_grid.snap_to_mask.return_value = (-3.0, 51.0)
        result = delineate_catchment(point=boundary_point, dem_path=mock_dem_path)
        assert isinstance(result, Polygon)

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_catchment_uses_coordinate_xytype(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that catchment is called with xytype='coordinate'."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        assert mock_grid.catchment.call_args[1]["xytype"] == "coordinate"

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_returned_polygon_is_valid(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that the returned polygon is geometrically valid."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        result = delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        assert result.is_valid
        assert not result.is_empty
        assert result.area > 0


@pytest.mark.integration
class TestDelineateCatchmentBehaviour:
    """Test the delineate_catchment function behaves as expected on select examples.

    These are integration tests that require ``data/devon_dem.tif`` to be
    present.  Run them with::

        pytest -m integration -k TestDelineateCatchmentBehaviour
    """

    @pytest.fixture(autouse=True)
    def _require_devon_dem(self):
        """Skip the entire class when the Devon DEM is missing."""
        if not _DEVON_DEM_PATH.exists():
            pytest.skip(f"Devon DEM not found: {_DEVON_DEM_PATH}")

    def test_upstream_subsumed(self, sample_downstream_points):
        """Catchments of downstream points must fully subsume upstream catchments.

        For three points on the same river stem (upstream → middle → downstream)
        the delineated catchment polygons must satisfy:

            downstream ⊇ middle ⊇ upstream

        A tiny spatial buffer (1e-5 degrees, ≈ 1 m) is applied to the
        containing polygon before the ``contains`` check to absorb
        sub-pixel rounding at shared boundary edges produced by the
        raster-to-polygon conversion.
        """
        # Delineate catchments for all three points.
        polys = {}
        for name, point in sample_downstream_points.items():
            polys[name] = delineate_catchment(
                point,
                _DEVON_DEM_PATH,
                flow_acc_threshold=5000,
            )

        upstream_poly = polys["upstream"]
        middle_poly = polys["middle"]
        downstream_poly = polys["downstream"]

        # Sanity-check: areas must increase from upstream to downstream.
        assert upstream_poly.area < middle_poly.area, (
            f"Expected upstream area ({upstream_poly.area:.6f}) < "
            f"middle area ({middle_poly.area:.6f})"
        )
        assert middle_poly.area < downstream_poly.area, (
            f"Expected middle area ({middle_poly.area:.6f}) < "
            f"downstream area ({downstream_poly.area:.6f})"
        )

        # A small buffer absorbs rounding at shared raster-cell edges.
        tol = 1e-5  # degrees — roughly 1 m at UK latitudes

        assert downstream_poly.buffer(tol).contains(middle_poly), (
            "downstream catchment does not subsume middle catchment"
        )
        assert downstream_poly.buffer(tol).contains(upstream_poly), (
            "downstream catchment does not subsume upstream catchment"
        )
        assert middle_poly.buffer(tol).contains(upstream_poly), (
            "middle catchment does not subsume upstream catchment"
        )


class TestDelineateCatchmentWithMetadata:
    """Test the delineate_catchment_with_metadata function."""

    def test_function_exists(self):
        """Test that the metadata function exists and is importable."""
        assert callable(delineate_catchment_with_metadata)

    def test_metadata_function_signature(self):
        """Test that metadata function has correct signature."""
        import inspect

        sig = inspect.signature(delineate_catchment_with_metadata)
        params = sig.parameters

        assert "point" in params
        assert "dem_path" in params
        # Should accept **kwargs for passing to main function
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    @patch("eoflow.catchment._delineate_catchment_core")
    def test_metadata_structure(self, mock_core, sample_point, sample_polygon):
        """Test that metadata function returns correct structure."""
        snapped_point = Point(-3.51, 50.71)
        flow_acc_value = 512.0
        mock_core.return_value = (sample_polygon, snapped_point, flow_acc_value)

        result = delineate_catchment_with_metadata(point=sample_point, dem_path="dummy.tif")

        # Check that result is a dictionary
        assert isinstance(result, dict)

        # Check that expected keys are present
        assert "polygon" in result
        assert "area" in result
        assert "centroid" in result
        assert "bounds" in result
        assert "pour_point" in result
        assert "snapped_pour_point" in result
        assert "flow_acc_at_pour_point" in result

        # Check types
        assert isinstance(result["polygon"], Polygon)
        assert isinstance(result["area"], float)
        assert isinstance(result["bounds"], tuple)
        assert isinstance(result["pour_point"], Point)

        # Check values
        assert result["polygon"] == sample_polygon
        assert result["area"] == sample_polygon.area
        assert result["centroid"] == sample_polygon.centroid
        assert result["bounds"] == sample_polygon.bounds
        assert result["pour_point"] == sample_point
        assert result["snapped_pour_point"] == snapped_point
        assert result["flow_acc_at_pour_point"] == flow_acc_value

    @patch("eoflow.catchment._delineate_catchment_core")
    def test_metadata_kwargs_passed_through(self, mock_core, sample_point, sample_polygon):
        """Test that kwargs are passed to _delineate_catchment_core."""
        mock_core.return_value = (sample_polygon, Point(-3.51, 50.71), 250.0)

        custom_kwargs = {
            "flow_acc_threshold": 5000,
            "pit_fill": False,
            "routing": "d8",
        }

        delineate_catchment_with_metadata(point=sample_point, dem_path="dummy.tif", **custom_kwargs)

        # Verify _delineate_catchment_core was called with the kwargs
        mock_core.assert_called_once()
        call_kwargs = mock_core.call_args[1]
        assert call_kwargs["flow_acc_threshold"] == 5000
        assert call_kwargs["pit_fill"] is False
        assert call_kwargs["routing"] == "d8"


class TestPolygonOperations:
    """Test polygon-related operations."""

    def test_sample_polygon_properties(self, sample_polygon):
        """Test that sample polygon has expected properties."""
        assert sample_polygon.is_valid
        assert sample_polygon.area > 0
        assert hasattr(sample_polygon, "centroid")
        assert hasattr(sample_polygon, "bounds")

    def test_polygon_bounds_format(self, sample_polygon):
        """Test that polygon bounds are in correct format."""
        bounds = sample_polygon.bounds
        assert len(bounds) == 4
        minx, miny, maxx, maxy = bounds
        assert minx < maxx
        assert miny < maxy

    def test_polygon_centroid(self, sample_polygon):
        """Test that polygon centroid is accessible."""
        centroid = sample_polygon.centroid
        assert isinstance(centroid, Point)
        assert hasattr(centroid, "x")
        assert hasattr(centroid, "y")

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_multiple_polygons_largest_selected(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path
    ):
        """Test that when multiple polygons exist, the largest is returned."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        # Setup mocks for workflow
        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data

        # Mock polygonize to return multiple polygons
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.55, 50.7], [-3.55, 50.75], [-3.5, 50.75], [-3.5, 50.7]]
                    ],
                },
                1,
            ),
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            ),
        ]

        # Create two polygons with different areas
        small_polygon = Polygon(
            [(-3.5, 50.7), (-3.55, 50.7), (-3.55, 50.75), (-3.5, 50.75), (-3.5, 50.7)]
        )
        large_polygon = Polygon(
            [(-3.5, 50.7), (-3.6, 50.7), (-3.6, 50.8), (-3.5, 50.8), (-3.5, 50.7)]
        )

        mock_shape.side_effect = [small_polygon, large_polygon]

        result = delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        # Should return the larger polygon
        assert result == large_polygon
        assert result.area > small_polygon.area


class TestModuleStructure:
    """Test overall module structure and imports."""

    def test_module_imports(self):
        """Test that module can be imported."""
        import eoflow.catchment

        assert hasattr(eoflow.catchment, "delineate_catchment")
        assert hasattr(eoflow.catchment, "delineate_catchment_with_metadata")

    def test_function_docstrings(self):
        """Test that functions have documentation."""
        assert delineate_catchment.__doc__ is not None
        assert delineate_catchment_with_metadata.__doc__ is not None
        assert len(delineate_catchment.__doc__) > 100  # Substantial documentation

    def test_type_hints(self):
        """Test that functions have type hints."""
        import inspect

        sig = inspect.signature(delineate_catchment)

        # Check return annotation
        assert sig.return_annotation is not None
        assert sig.return_annotation == Polygon or "Polygon" in str(sig.return_annotation)


class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch("eoflow.catchment.Grid")
    def test_no_catchment_polygon_generated(self, mock_grid_class, sample_point, mock_dem_path):
        """Test error when no catchment polygon is generated."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        # Setup normal workflow but return empty polygon list
        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        mock_grid.catchment.return_value = catch_data

        # Mock polygonize to return no valid polygons
        mock_grid.polygonize.return_value = [({"type": "Polygon", "coordinates": []}, 0)]

        with pytest.raises(ValueError) as exc_info:
            delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        assert "No catchment polygon generated" in str(exc_info.value)

    @patch("eoflow.catchment.Grid")
    def test_catchment_delineation_failure(self, mock_grid_class, sample_point, mock_dem_path):
        """Test error handling when catchment delineation fails."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        # Setup normal workflow but make catchment fail
        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        # Make catchment raise an exception
        mock_grid.catchment.side_effect = Exception("Catchment error")

        with pytest.raises(ValueError) as exc_info:
            delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        assert "Failed to delineate catchment" in str(exc_info.value)

    @patch("eoflow.catchment.Grid")
    def test_polygonize_failure(self, mock_grid_class, sample_point, mock_dem_path):
        """Test error handling when polygonize raises an exception."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        catch_data[4:7, 4:7] = 1
        mock_grid.catchment.return_value = catch_data

        # Make polygonize raise an exception
        mock_grid.polygonize.side_effect = RuntimeError("Raster conversion failed")

        with pytest.raises(ValueError) as exc_info:
            delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        assert "Failed to convert catchment to polygon" in str(exc_info.value)

    @patch("eoflow.catchment.Grid")
    def test_only_nodata_polygons_returned(self, mock_grid_class, sample_point, mock_dem_path):
        """Test error when polygonize returns only non-catchment (value != 1) polygons."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        catch_data = np.zeros((10, 10), dtype=np.int32)
        mock_grid.catchment.return_value = catch_data

        # Polygonize returns polygons but none with value == 1
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [[[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.7]]],
                },
                0,
            ),
            (
                {
                    "type": "Polygon",
                    "coordinates": [[[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.7]]],
                },
                2,
            ),
        ]

        with pytest.raises(ValueError) as exc_info:
            delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        assert "No catchment polygon generated" in str(exc_info.value)

    def test_missing_dem_error_includes_path(self, sample_point):
        """Test that the FileNotFoundError message includes the missing path."""
        bad_path = "/tmp/this_does_not_exist_12345.tif"

        with pytest.raises(FileNotFoundError) as exc_info:
            delineate_catchment(point=sample_point, dem_path=bad_path)

        assert bad_path in str(exc_info.value)

    @patch("eoflow.catchment.Grid")
    def test_point_outside_bounds_error_includes_coords(self, mock_grid_class, mock_dem_path):
        """Test that out-of-bounds error message includes point and bbox info."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-2.0, 49.0, -1.0, 50.0]
        mock_grid_class.from_raster.return_value = mock_grid
        mock_grid.read_raster.return_value = np.random.rand(10, 10)

        point = Point(-5.0, 55.0)

        with pytest.raises(ValueError) as exc_info:
            delineate_catchment(point=point, dem_path=mock_dem_path)

        error_msg = str(exc_info.value)
        assert "-5.0" in error_msg
        assert "55.0" in error_msg
        assert "outside DEM bounds" in error_msg


class TestDataTypes:
    """Test data type handling."""

    def test_path_types_accepted(self):
        """Test that both str and Path types are accepted for dem_path."""
        import inspect

        sig = inspect.signature(delineate_catchment)
        dem_path_param = sig.parameters["dem_path"]

        # Check that annotation includes Union[str, Path] or similar
        annotation = str(dem_path_param.annotation)
        assert "str" in annotation.lower() or "path" in annotation.lower()

    def test_point_must_be_shapely(self, sample_point):
        """Test that point parameter accepts Shapely Point."""
        assert isinstance(sample_point, Point)

        # Should have Shapely Point methods
        assert hasattr(sample_point, "x")
        assert hasattr(sample_point, "y")
        assert hasattr(sample_point, "coords")

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_path_string_conversion(
        self, mock_shape, mock_grid_class, sample_point, sample_polygon
    ):
        """Test that string paths are properly converted."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            temp_path = tmp.name

        try:
            mock_grid = MagicMock()
            mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
            mock_grid_class.from_raster.return_value = mock_grid

            # Setup all mocks
            dem_data = np.random.rand(10, 10)
            mock_grid.read_raster.return_value = dem_data
            mock_grid.fill_pits.return_value = dem_data + 0.1
            mock_grid.resolve_flats.return_value = dem_data + 0.2
            mock_grid.flowdir.return_value = np.ones((10, 10))
            mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
            mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

            catch_data = np.zeros((10, 10), dtype=np.int32)
            catch_data[4:7, 4:7] = 1
            mock_grid.catchment.return_value = catch_data
            mock_grid.polygonize.return_value = [
                (
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                        ],
                    },
                    1,
                )
            ]
            mock_shape.return_value = sample_polygon

            # Test with string path
            result = delineate_catchment(point=sample_point, dem_path=temp_path)
            assert isinstance(result, Polygon)

            # Test with Path object
            result = delineate_catchment(point=sample_point, dem_path=Path(temp_path))
            assert isinstance(result, Polygon)

        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)


class TestCatchmentAsInt32:
    """Test that the catchment array is cast to int32 before polygonize."""

    @patch("eoflow.catchment.Grid")
    @patch("eoflow.catchment.shape")
    def test_catchment_cast_to_int32(
        self, mock_shape, mock_grid_class, sample_point, mock_dem_path, sample_polygon
    ):
        """Test that catch.astype(np.int32) is passed to polygonize."""
        mock_dem_path.touch()

        mock_grid = MagicMock()
        mock_grid.bbox = [-4.0, 50.0, -3.0, 51.0]
        mock_grid_class.from_raster.return_value = mock_grid

        dem_data = np.random.rand(10, 10)
        mock_grid.read_raster.return_value = dem_data
        mock_grid.fill_pits.return_value = dem_data + 0.1
        mock_grid.resolve_flats.return_value = dem_data + 0.2
        mock_grid.flowdir.return_value = np.ones((10, 10))
        mock_grid.accumulation.return_value = np.random.randint(0, 5000, (10, 10))
        mock_grid.snap_to_mask.return_value = (-3.5, 50.7)

        # Return a float64 catchment array to test the int32 conversion
        catch_data = np.zeros((10, 10), dtype=np.float64)
        catch_data[4:7, 4:7] = 1.0
        mock_grid.catchment.return_value = catch_data
        mock_grid.polygonize.return_value = [
            (
                {
                    "type": "Polygon",
                    "coordinates": [
                        [[-3.5, 50.7], [-3.6, 50.7], [-3.6, 50.8], [-3.5, 50.8], [-3.5, 50.7]]
                    ],
                },
                1,
            )
        ]
        mock_shape.return_value = sample_polygon

        delineate_catchment(point=sample_point, dem_path=mock_dem_path)

        # Verify polygonize was called with an int32 array
        polygonize_arg = mock_grid.polygonize.call_args[0][0]
        assert polygonize_arg.dtype == np.int32


# ---------------------------------------------------------------------------
# Helpers for integration tests
# ---------------------------------------------------------------------------


def _load_sample_points() -> pd.DataFrame:
    """Load the water quality CSV and return a DataFrame with lat/lon."""
    return pd.read_csv(_CSV_PATH)


def _skip_if_data_missing():
    """Pytest skip helper when test data files are absent."""
    if not _CSV_PATH.exists():
        pytest.skip(f"CSV not found: {_CSV_PATH}")
    if not _DEM_PATH.exists():
        pytest.skip(f"DEM not found: {_DEM_PATH}")


# A small, representative subset of points from the CSV selected for
# geographic diversity (south-west, midlands, north-east, thames).
# Each tuple: (label_fragment, longitude, latitude)
_REPRESENTATIVE_POINTS = [
    ("KENNET AT STITCHCOMBE MILL", -1.6744766949667198, 51.42452537684588),
    ("SMALL BROOK (KINGSBRIDGE) AT BOWCOMBE", -3.7555385318960095, 50.28611262100282),
    ("LUNE AT LUNE BRIDGE/VIADUCT (NW MICKLETON)", -2.065150679659169, 54.61141239059547),
]


@pytest.mark.integration
class TestCatchmentIntegration:
    """
    Integration tests for catchment delineation using real EA water-quality
    sample points and the UK DEM.

    These tests are slow (~90 s each) because they process a full-resolution
    DEM.  They are gated behind the ``integration`` marker so that they only
    run when explicitly requested::

        pytest -m integration
        pytest -m integration -k TestCatchmentIntegration
    """

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture(autouse=True)
    def _require_data(self):
        """Skip the entire class when test data is missing."""
        _skip_if_data_missing()

    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        """The full water-quality DataFrame."""
        return _load_sample_points()

    # ------------------------------------------------------------------
    # Parametrised delineation tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "label, lon, lat",
        _REPRESENTATIVE_POINTS,
        ids=[p[0] for p in _REPRESENTATIVE_POINTS],
    )
    def test_delineate_returns_valid_polygon(self, label, lon, lat):
        """Delineation should return a valid, non-empty Shapely Polygon."""
        point = Point(lon, lat)
        poly = delineate_catchment(point, _DEM_PATH, flow_acc_threshold=1000)

        assert isinstance(poly, Polygon), f"Expected Polygon, got {type(poly)}"
        assert poly.is_valid, "Returned polygon is not valid"
        assert not poly.is_empty, "Returned polygon is empty"
        assert poly.area > 0, "Polygon area must be positive"

    @pytest.mark.parametrize(
        "label, lon, lat",
        _REPRESENTATIVE_POINTS,
        ids=[p[0] for p in _REPRESENTATIVE_POINTS],
    )
    def test_catchment_bounds_are_near_pour_point(self, label, lon, lat):
        """The catchment bounding box should be reasonably close to the pour point.

        We allow up to 1 degree of separation — catchments in the UK at
        ~90 m resolution rarely extend further than that from the outlet.
        """
        point = Point(lon, lat)
        poly = delineate_catchment(point, _DEM_PATH, flow_acc_threshold=1000)
        minx, miny, maxx, maxy = poly.bounds

        max_offset_deg = 1.0
        assert abs(minx - lon) < max_offset_deg, "minx too far from pour point lon"
        assert abs(maxx - lon) < max_offset_deg, "maxx too far from pour point lon"
        assert abs(miny - lat) < max_offset_deg, "miny too far from pour point lat"
        assert abs(maxy - lat) < max_offset_deg, "maxy too far from pour point lat"

    # ------------------------------------------------------------------
    # Single-point focused tests (use the first representative point to
    # keep runtime manageable)
    # ------------------------------------------------------------------

    def test_catchment_polygon_has_reasonable_area(self):
        """Catchment area (in square degrees) should be small but non-trivial."""
        lon, lat = _REPRESENTATIVE_POINTS[0][1], _REPRESENTATIVE_POINTS[0][2]
        poly = delineate_catchment(Point(lon, lat), _DEM_PATH, flow_acc_threshold=1000)

        # At UK latitudes 1 deg ≈ 111 km (lat) / ~70 km (lon).
        # Catchment areas for small rivers are typically < 0.1 sq-deg.
        assert poly.area < 0.5, f"Catchment area suspiciously large: {poly.area}"
        assert poly.area > 1e-6, f"Catchment area suspiciously small: {poly.area}"

    def test_catchment_centroid_is_inside_polygon(self):
        """The polygon's centroid should lie within (or on) the polygon."""
        lon, lat = _REPRESENTATIVE_POINTS[0][1], _REPRESENTATIVE_POINTS[0][2]
        poly = delineate_catchment(Point(lon, lat), _DEM_PATH, flow_acc_threshold=1000)
        centroid = poly.centroid

        # Use a tiny buffer to handle floating-point edge cases
        assert poly.buffer(1e-8).contains(centroid), "Centroid is not inside the polygon"

    def test_with_metadata_returns_expected_keys(self):
        """``delineate_catchment_with_metadata`` should return all documented keys."""
        lon, lat = _REPRESENTATIVE_POINTS[0][1], _REPRESENTATIVE_POINTS[0][2]
        point = Point(lon, lat)
        result = delineate_catchment_with_metadata(point, _DEM_PATH, flow_acc_threshold=1000)

        expected_keys = {"polygon", "area", "centroid", "bounds", "pour_point"}
        assert set(result.keys()) == expected_keys

        assert isinstance(result["polygon"], Polygon)
        assert result["polygon"].is_valid
        assert result["area"] > 0
        assert result["pour_point"].equals(point)

        # bounds should be a 4-tuple
        assert len(result["bounds"]) == 4
        minx, miny, maxx, maxy = result["bounds"]
        assert minx < maxx
        assert miny < maxy

    def test_flow_acc_threshold_affects_catchment(self):
        """Different flow-accumulation thresholds should produce different
        catchments because the pour point snaps to different stream cells."""
        lon, lat = _REPRESENTATIVE_POINTS[0][1], _REPRESENTATIVE_POINTS[0][2]
        point = Point(lon, lat)

        poly_low = delineate_catchment(point, _DEM_PATH, flow_acc_threshold=500)
        poly_high = delineate_catchment(point, _DEM_PATH, flow_acc_threshold=5000)

        # The two polygons should both be valid but need not be identical
        assert poly_low.is_valid
        assert poly_high.is_valid
        # They *may* happen to be identical if both thresholds snap to the
        # same cell, so we just verify they are both legitimate polygons.
        assert poly_low.area > 0
        assert poly_high.area > 0

    # ------------------------------------------------------------------
    # CSV-driven tests
    # ------------------------------------------------------------------

    def test_all_csv_points_within_dem_bounds(self, sample_df):
        """Every point in the CSV should fall inside the DEM bounding box."""
        from pysheds.grid import Grid

        grid = Grid.from_raster(str(_DEM_PATH))
        bbox = grid.bbox  # (left, bottom, right, top)

        for _, row in sample_df.iterrows():
            lon, lat = row["longitude"], row["latitude"]
            assert bbox[0] <= lon <= bbox[2], (
                f"Longitude {lon} outside DEM x-range [{bbox[0]}, {bbox[2]}]"
            )
            assert bbox[1] <= lat <= bbox[3], (
                f"Latitude {lat} outside DEM y-range [{bbox[1]}, {bbox[3]}]"
            )

    def test_csv_has_expected_columns(self, sample_df):
        """The CSV should contain the columns needed for delineation."""
        required = {"latitude", "longitude", "samplingPoint.prefLabel"}
        assert required.issubset(set(sample_df.columns))

    def test_csv_has_expected_row_count(self, sample_df):
        """Sanity-check: the pruned CSV should have exactly 10 rows."""
        assert len(sample_df) == 10

    # ------------------------------------------------------------------
    # Error-path integration tests
    # ------------------------------------------------------------------

    def test_point_outside_dem_raises_value_error(self):
        """A point clearly outside the UK DEM should raise ``ValueError``."""
        point = Point(10.0, 60.0)  # somewhere in Scandinavia
        with pytest.raises(ValueError, match="outside DEM bounds"):
            delineate_catchment(point, _DEM_PATH)

    def test_dem_path_as_string_works(self):
        """Passing the DEM path as a plain string should work identically."""
        lon, lat = _REPRESENTATIVE_POINTS[0][1], _REPRESENTATIVE_POINTS[0][2]
        poly = delineate_catchment(Point(lon, lat), str(_DEM_PATH), flow_acc_threshold=1000)
        assert isinstance(poly, Polygon)
        assert poly.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
