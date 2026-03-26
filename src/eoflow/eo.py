"""Core earth-observation helpers built on top of openEO.

This module provides low-level building blocks for constructing, computing,
and materialising Sentinel-2 datacubes via the openEO API.  The
:class:`~eoflow.samples.Sample` class uses these helpers for its higher-level
``query_*`` and ``fetch_*`` methods, but every function here can also be
called directly for custom workflows.

Typical usage
-------------
::

    import eoflow.eo as eo

    conn = eo.connect()                          # authenticate via OIDC

    spatial_extent = {"west": -3.6, "south": 50.6, "east": -3.4, "north": 50.8}
    temporal_extent = ("2024-03-01", "2024-04-30")

    # Build and download NDVI
    cube   = eo.build_index_cube(conn, "NDVI", spatial_extent, temporal_extent)
    ds     = eo.download_cube_as_xarray(cube)
    ndvi   = eo.extract_dataarray(ds, name="ndvi")

    # Or work band-by-band
    cube   = eo.build_band_cube(conn, spatial_extent, temporal_extent, ["B04", "B08"])
    ds     = eo.download_cube_as_xarray(cube)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

from eoflow.log_utils import get_logger

if TYPE_CHECKING:
    import openeo  # type: ignore[import-untyped]
    import xarray  # type: ignore[import-untyped]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default openEO backend URL (Copernicus Data Space Federation)
OPENEO_BACKEND: str = "openeofed.dataspace.copernicus.eu"

#: Default Sentinel-2 L2A collection identifier on that backend
SENTINEL2_COLLECTION: str = "SENTINEL2_L2A"

#: Band names required by each named spectral index on SENTINEL2_L2A.
#: Keys are upper-case index names; values are dicts mapping semantic role
#: (e.g. ``"nir"``, ``"red"``) to the actual band identifier.
S2_INDEX_BANDS: Dict[str, Dict[str, str]] = {
    "NDVI": {"nir": "B08", "red": "B04"},
    "NDWI": {"green": "B03", "nir": "B08"},
    "EVI": {"nir": "B08", "red": "B04", "blue": "B02"},
    "MNDWI": {"green": "B03", "swir": "B11"},
    "NDRE": {"rededge": "B05", "red": "B04"},
}

#: Tuple of all supported spectral index names (upper-case).
SUPPORTED_INDICES: Tuple[str, ...] = tuple(S2_INDEX_BANDS.keys())


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def connect(
    backend: str = OPENEO_BACKEND,
    *,
    authenticate: bool = True,
) -> "openeo.Connection":
    """Connect to an openEO backend and optionally authenticate via OIDC.

    Parameters
    ----------
    backend : str
        URL of the openEO backend.  Defaults to the Copernicus Data Space
        Federation endpoint (``openeofed.dataspace.copernicus.eu``).
    authenticate : bool
        If *True* (default), trigger OIDC device-flow authentication.
        Pass *False* for an unauthenticated connection (only publicly
        accessible collections will be accessible).

    Returns
    -------
    openeo.Connection

    Raises
    ------
    ImportError
        If the ``openeo`` package is not installed.

    Examples
    --------
    ::

        conn = eo.connect()                       # interactive OIDC login
        conn = eo.connect(authenticate=False)     # no auth (public only)
    """
    try:
        import openeo as _openeo
    except ImportError as exc:
        raise ImportError(
            "The 'openeo' package is required for EO queries.  "
            "Install it with:  uv add openeo  (or  pip install openeo)"
        ) from exc

    logger.info("Connecting to openEO backend: %s", backend)
    conn = _openeo.connect(backend)
    if authenticate:
        conn.authenticate_oidc()
        logger.info("OIDC authentication complete")
    return conn


# ---------------------------------------------------------------------------
# Datacube construction
# ---------------------------------------------------------------------------


def build_band_cube(
    connection: "openeo.Connection",
    spatial_extent: Dict[str, float],
    temporal_extent: Tuple[str, str],
    bands: Sequence[str],
    *,
    collection: str = SENTINEL2_COLLECTION,
    max_cloud_cover: int = 85,
) -> "openeo.DataCube":
    """Load a Sentinel-2 datacube for the specified bands and extents.

    This is a thin wrapper around ``connection.load_collection`` that adds
    logging and enforces consistent argument types.

    Parameters
    ----------
    connection : openeo.Connection
        An authenticated (or public) openEO connection.
    spatial_extent : dict
        Bounding-box dict with keys ``west``, ``south``, ``east``, ``north``
        in WGS 84 decimal degrees.
    temporal_extent : tuple of str
        ``(start_date, end_date)`` as ISO-8601 strings (``"YYYY-MM-DD"``).
    bands : sequence of str
        Band names to load, e.g. ``["B04", "B08"]``.
    collection : str
        openEO collection ID.  Defaults to ``SENTINEL2_L2A``.
    max_cloud_cover : int
        Maximum cloud cover percentage (0–100).  Scenes above this threshold
        are pre-filtered before the process graph runs.

    Returns
    -------
    openeo.DataCube
        A lazy process-graph node.  Call :func:`download_cube_as_xarray` (or
        the openEO batch-job API) to materialise it.

    Examples
    --------
    ::

        cube = eo.build_band_cube(
            conn,
            spatial_extent={"west": -3.6, "south": 50.6, "east": -3.4, "north": 50.8},
            temporal_extent=("2024-03-01", "2024-04-30"),
            bands=["B04", "B08"],
        )
    """
    logger.info(
        "Loading %s | bands=%s | %s → %s",
        collection,
        list(bands),
        temporal_extent[0],
        temporal_extent[1],
    )
    return connection.load_collection(
        collection,
        spatial_extent=spatial_extent,
        temporal_extent=list(temporal_extent),
        bands=list(bands),
        max_cloud_cover=max_cloud_cover,
    )


# ---------------------------------------------------------------------------
# Spectral index computations
# (each function accepts an existing DataCube and returns a new one)
# ---------------------------------------------------------------------------


def compute_ndvi(
    cube: "openeo.DataCube",
    *,
    nir_band: str = "B08",
    red_band: str = "B04",
) -> "openeo.DataCube":
    """Apply NDVI = (NIR − Red) / (NIR + Red) to a pre-loaded datacube.

    Uses the openEO built-in ``ndvi`` process, which is optimised on most
    backends.

    Parameters
    ----------
    cube : openeo.DataCube
        Must already contain ``nir_band`` and ``red_band``.
    nir_band : str
        Name of the NIR band (default ``"B08"`` — Sentinel-2 10 m).
    red_band : str
        Name of the red band (default ``"B04"`` — Sentinel-2 10 m).

    Returns
    -------
    openeo.DataCube
        Single-band cube with NDVI values in the range [−1, 1].
    """
    ndvi = cube.ndvi(nir=nir_band, red=red_band)
    logger.debug("NDVI applied (nir=%s, red=%s)", nir_band, red_band)
    return ndvi


def compute_ndwi(
    cube: "openeo.DataCube",
    *,
    green_band: str = "B03",
    nir_band: str = "B08",
) -> "openeo.DataCube":
    """Apply NDWI = (Green − NIR) / (Green + NIR) to a pre-loaded datacube.

    The Normalised Difference Water Index (McFeeters 1996) highlights open
    water surfaces.  openEO has no built-in ``ndwi`` process, so this is
    implemented via band arithmetic.

    Parameters
    ----------
    cube : openeo.DataCube
        Must already contain ``green_band`` and ``nir_band``.
    green_band : str
        Name of the green band (default ``"B03"`` — Sentinel-2 10 m).
    nir_band : str
        Name of the NIR band (default ``"B08"`` — Sentinel-2 10 m).

    Returns
    -------
    openeo.DataCube
        Single-band cube with NDWI values in the range [−1, 1].
    """
    green = cube.filter_bands([green_band]).rename_labels(dimension="bands", target=["green"])
    nir = cube.filter_bands([nir_band]).rename_labels(dimension="bands", target=["nir"])
    merged = green.merge_cubes(nir)
    ndwi = merged.reduce_dimension(
        dimension="bands",
        reducer=lambda data: (
            (data.array_element(0) - data.array_element(1))
            / (data.array_element(0) + data.array_element(1))
        ),
    )
    logger.debug("NDWI applied (green=%s, nir=%s)", green_band, nir_band)
    return ndwi


def compute_normalised_difference(
    cube: "openeo.DataCube",
    band_a: str,
    band_b: str,
    label: str = "ND",
) -> "openeo.DataCube":
    """Apply (band_a − band_b) / (band_a + band_b) to a pre-loaded datacube.

    A generic helper for any two-band normalised difference index (MNDWI,
    NDRE, custom indices, etc.).

    Parameters
    ----------
    cube : openeo.DataCube
        Must already contain ``band_a`` and ``band_b``.
    band_a : str
        First band identifier (appears in the numerator as ``band_a − band_b``).
    band_b : str
        Second band identifier.
    label : str
        Descriptive label used in log messages (e.g. ``"MNDWI"``).

    Returns
    -------
    openeo.DataCube
        Single-band cube with normalised-difference values in [−1, 1].
    """
    a = cube.filter_bands([band_a]).rename_labels(dimension="bands", target=["a"])
    b = cube.filter_bands([band_b]).rename_labels(dimension="bands", target=["b"])
    merged = a.merge_cubes(b)
    result = merged.reduce_dimension(
        dimension="bands",
        reducer=lambda data: (
            (data.array_element(0) - data.array_element(1))
            / (data.array_element(0) + data.array_element(1))
        ),
    )
    logger.debug("%s applied (%s, %s)", label, band_a, band_b)
    return result


def compute_evi(
    cube: "openeo.DataCube",
    *,
    nir_band: str = "B08",
    red_band: str = "B04",
    blue_band: str = "B02",
) -> "openeo.DataCube":
    """Apply EVI = 2.5 * (NIR − Red) / (NIR + 6*Red − 7.5*Blue + 1).

    The Enhanced Vegetation Index (Huete et al. 1997/2002) corrects for
    atmospheric and canopy background effects using a three-band formula.

    Notes
    -----
    Sentinel-2 L2A reflectance is scaled ×10 000.  The absolute ``+1`` term
    in the standard formula is expressed as ``+10 000`` here so that it is
    commensurate with the scaled reflectance values, giving results that are
    numerically identical to applying the formula to [0, 1] reflectance.

    Parameters
    ----------
    cube : openeo.DataCube
        Must already contain ``nir_band``, ``red_band``, and ``blue_band``.
    nir_band : str
        Default ``"B08"`` (Sentinel-2 10 m NIR).
    red_band : str
        Default ``"B04"`` (Sentinel-2 10 m red).
    blue_band : str
        Default ``"B02"`` (Sentinel-2 10 m blue).

    Returns
    -------
    openeo.DataCube
        Single-band cube with EVI values (typically in the range [−1, 1],
        though atmospheric artefacts can push values slightly outside this).
    """
    nir = cube.filter_bands([nir_band]).rename_labels(dimension="bands", target=["nir"])
    red = cube.filter_bands([red_band]).rename_labels(dimension="bands", target=["red"])
    blue = cube.filter_bands([blue_band]).rename_labels(dimension="bands", target=["blue"])
    merged = nir.merge_cubes(red).merge_cubes(blue)

    def _evi_reducer(data):
        nir_v = data.array_element(0)
        red_v = data.array_element(1)
        blue_v = data.array_element(2)
        # +10 000 in the denominator accounts for the ×10 000 L2A scale factor
        return 2.5 * (nir_v - red_v) / (nir_v + 6 * red_v - 7.5 * blue_v + 10000)

    evi = merged.reduce_dimension(dimension="bands", reducer=_evi_reducer)
    logger.debug("EVI applied (nir=%s, red=%s, blue=%s)", nir_band, red_band, blue_band)
    return evi


# ---------------------------------------------------------------------------
# Named-index dispatch
# ---------------------------------------------------------------------------


def build_index_cube(
    connection: "openeo.Connection",
    index: str,
    spatial_extent: Dict[str, float],
    temporal_extent: Tuple[str, str],
    *,
    collection: str = SENTINEL2_COLLECTION,
    max_cloud_cover: int = 85,
) -> "openeo.DataCube":
    """Build a complete datacube for a named spectral index.

    Combines :func:`build_band_cube` with the appropriate ``compute_*``
    function for the requested index.

    Supported index names (case-insensitive):

    +----------+--------------------------------------------+
    | Name     | Formula                                    |
    +==========+============================================+
    | NDVI     | (NIR − Red) / (NIR + Red)                  |
    +----------+--------------------------------------------+
    | NDWI     | (Green − NIR) / (Green + NIR)              |
    +----------+--------------------------------------------+
    | MNDWI    | (Green − SWIR) / (Green + SWIR)            |
    +----------+--------------------------------------------+
    | NDRE     | (RedEdge − Red) / (RedEdge + Red)          |
    +----------+--------------------------------------------+
    | EVI      | 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1)      |
    +----------+--------------------------------------------+

    Parameters
    ----------
    connection : openeo.Connection
    index : str
        Spectral index name (see table above).  Case-insensitive.
    spatial_extent : dict
        Bounding-box dict (``west``, ``south``, ``east``, ``north``).
    temporal_extent : tuple of str
        ``(start_date, end_date)`` as ISO-8601 strings.
    collection : str
        openEO collection ID.  Defaults to ``SENTINEL2_L2A``.
    max_cloud_cover : int
        Maximum cloud cover percentage (0–100).

    Returns
    -------
    openeo.DataCube
        A lazy single-band datacube for the requested index.

    Raises
    ------
    ValueError
        If *index* is not one of the supported names.

    Examples
    --------
    ::

        ndvi_cube = eo.build_index_cube(
            conn, "NDVI",
            spatial_extent={"west": -3.6, "south": 50.6, "east": -3.4, "north": 50.8},
            temporal_extent=("2024-03-01", "2024-04-30"),
        )
        ds = eo.download_cube_as_xarray(ndvi_cube)
    """
    idx = index.upper()
    if idx not in S2_INDEX_BANDS:
        raise ValueError(
            f"Unknown index '{index}'.  "
            f"Supported indices: {', '.join(sorted(SUPPORTED_INDICES))}.  "
            "For arbitrary bands use build_band_cube() instead."
        )

    logger.info("Building %s cube | %s → %s", idx, temporal_extent[0], temporal_extent[1])

    if idx == "NDVI":
        b = S2_INDEX_BANDS["NDVI"]
        cube = build_band_cube(
            connection,
            spatial_extent,
            temporal_extent,
            [b["nir"], b["red"]],
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        return compute_ndvi(cube, nir_band=b["nir"], red_band=b["red"])

    if idx == "NDWI":
        b = S2_INDEX_BANDS["NDWI"]
        cube = build_band_cube(
            connection,
            spatial_extent,
            temporal_extent,
            [b["green"], b["nir"]],
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        return compute_ndwi(cube, green_band=b["green"], nir_band=b["nir"])

    if idx == "MNDWI":
        b = S2_INDEX_BANDS["MNDWI"]
        cube = build_band_cube(
            connection,
            spatial_extent,
            temporal_extent,
            [b["green"], b["swir"]],
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        return compute_normalised_difference(cube, b["green"], b["swir"], "MNDWI")

    if idx == "NDRE":
        b = S2_INDEX_BANDS["NDRE"]
        cube = build_band_cube(
            connection,
            spatial_extent,
            temporal_extent,
            [b["rededge"], b["red"]],
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        return compute_normalised_difference(cube, b["rededge"], b["red"], "NDRE")

    if idx == "EVI":
        b = S2_INDEX_BANDS["EVI"]
        cube = build_band_cube(
            connection,
            spatial_extent,
            temporal_extent,
            [b["nir"], b["red"], b["blue"]],
            collection=collection,
            max_cloud_cover=max_cloud_cover,
        )
        return compute_evi(cube, nir_band=b["nir"], red_band=b["red"], blue_band=b["blue"])

    # Guard — should be unreachable given the membership check above
    raise ValueError(f"Unhandled index '{index}'")  # pragma: no cover


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------


def download_cube_as_xarray(
    cube: "openeo.DataCube",
    *,
    tmp_dir: Optional[Path] = None,
) -> "xarray.Dataset":
    """Execute a datacube synchronously and return the result as an xarray Dataset.

    The cube is downloaded to a temporary NetCDF file (chosen because NetCDF
    preserves full dimension metadata), loaded fully into memory, and the
    temporary file is then removed.

    Parameters
    ----------
    cube : openeo.DataCube
        A fully-specified openEO process graph ready for synchronous
        execution.  The cube should cover a reasonably small spatial extent
        and/or time range — for very large requests use the openEO batch-job
        API (``cube.execute_batch()``) instead.
    tmp_dir : Path, optional
        Directory to write the temporary download file in.  Defaults to the
        system temporary directory.

    Returns
    -------
    xarray.Dataset
        The result loaded fully into memory (``ds.load()`` called before the
        temp file is deleted).  Dimension and coordinate names reflect
        whatever the backend writes into the NetCDF.

    Raises
    ------
    ImportError
        If the ``xarray`` package is not installed.
    RuntimeError
        If the openEO backend returns an error during synchronous execution.

    Notes
    -----
    For a single-band result (e.g. NDVI), use :func:`extract_dataarray` to
    unwrap the Dataset into a :class:`xarray.DataArray`.

    Examples
    --------
    ::

        cube = eo.build_index_cube(conn, "NDVI", extent, dates)
        ds   = eo.download_cube_as_xarray(cube)
        ndvi = eo.extract_dataarray(ds, name="ndvi")
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "The 'xarray' package is required to materialise EO results.  "
            "Install it with:  uv add xarray  (or  pip install xarray)"
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".nc", dir=tmp_dir, delete=False) as f:
        tmp_path = Path(f.name)

    try:
        logger.info("Executing datacube — downloading to temporary NetCDF …")
        cube.download(str(tmp_path), format="netCDF")
        logger.debug("Download complete; opening with xarray …")
        ds = xr.open_dataset(tmp_path)
        ds.load()  # pull fully into memory before the temp file is deleted
        logger.info(
            "Datacube materialised: dims=%s, vars=%s",
            dict(ds.dims),
            list(ds.data_vars),
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return ds


# ---------------------------------------------------------------------------
# xarray helpers
# ---------------------------------------------------------------------------


def extract_dataarray(
    ds: "xarray.Dataset",
    name: Optional[str] = None,
) -> "xarray.DataArray":
    """Extract a single :class:`xarray.DataArray` from a Dataset.

    When *ds* contains exactly one data variable that DataArray is returned
    directly (optionally renamed to *name*).  When there are multiple
    variables they are stacked along a new ``"variable"`` dimension and any
    length-1 ``"variable"`` dimension is squeezed away.

    This is the recommended way to convert the output of
    :func:`download_cube_as_xarray` for a single-band result (e.g. NDVI,
    NDWI) into a plain DataArray for storage in
    :class:`~eoflow.samples.CatchmentLayers`.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset returned by :func:`download_cube_as_xarray`.
    name : str, optional
        If provided, the resulting DataArray is renamed to *name*.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    ::

        ds   = eo.download_cube_as_xarray(ndvi_cube)
        ndvi = eo.extract_dataarray(ds, name="ndvi")
        # ndvi is an xarray.DataArray with dims like (t, y, x)
    """
    if not ds.data_vars:
        raise ValueError(
            "Dataset has no data variables — cannot extract a DataArray. "
            "This usually means the openEO backend returned an empty result "
            "(e.g. all pixels were masked by the cloud-cover filter, or the "
            "spatial/temporal extent produced no matching scenes)."
        )

    # Strip known CF-convention CRS / spatial-reference metadata variables
    # that the CDSE / Terrascope openEO backend writes alongside the real data
    # (e.g. a scalar 'crs' variable).  If we left them in, to_array() would
    # stack them with the real data and produce a spurious 'variable' dimension.
    _META_VAR_NAMES = frozenset({"crs", "spatial_ref", "crs_wkt"})
    real_vars = {k: v for k, v in ds.data_vars.items() if k not in _META_VAR_NAMES}
    if not real_vars:
        real_vars = dict(ds.data_vars)  # fall back to all vars if everything was filtered

    if len(real_vars) == 1:
        var_key = next(iter(real_vars))
        da: "xarray.DataArray" = ds[var_key]
    else:
        da = ds[list(real_vars)].to_array(dim="variable")
        if da.sizes.get("variable", 0) == 1:
            da = da.squeeze("variable", drop=True)

    if name is not None:
        da = da.rename(name)

    # ── Coerce to float32 ────────────────────────────────────────────────────
    # Some openEO backends (e.g. CDSE / Terrascope) return object-dtype arrays
    # where NaN is encoded as b'' (empty bytestring).  Cast here so that the
    # DataArray stored in Sample.layers and written to NetCDF is always numeric.
    #
    # Note: numpy understands byte-string encoded numbers (np.float32(b'0.75')
    # → 0.75) but Python's built-in float() does not, so we use np.float32.
    import numpy as _np  # noqa: PLC0415

    if not _np.issubdtype(da.dtype, _np.floating):
        raw = da.values
        if raw.dtype.kind == "O":

            def _to_f(v: object) -> float:
                if v in (b"", b"nan", "", None):
                    return _np.nan
                try:
                    return float(_np.float32(v))  # type: ignore[arg-type]
                except (ValueError, TypeError):
                    return _np.nan

            da = da.copy(data=_np.vectorize(_to_f)(raw).astype(_np.float32))
            logger.debug(
                "extract_dataarray: coerced object-dtype array to float32 "
                "(backend encoded NaN as b'')"
            )
        else:
            da = da.astype(_np.float32)

    return da
