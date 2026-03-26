"""
soil.py — DEFRA Soil Structure Risk Groups: query and catchment coverage analysis.

This module queries the Environment Agency / DEFRA **Soil Erosion and Runoff
Groups (SEARG)** dataset via the ArcGIS REST Feature Service and computes
the fractional coverage of each of the 12 soil-structure groups within a
given catchment polygon.

Source dataset
--------------
    Title   : DefraSoilStructureGroups20240617
    Item    : https://www.arcgis.com/home/item.html?id=af498634e7c2409c8a0e3eecb720f8dc
    Service : https://services1.arcgis.com/JZM7qJpmv7vJ0Hzx/arcgis/rest/services/
              SEARGFull_DEFRA12SoilGroups20240617/FeatureServer/0
    Licence : Cranfield University LandIS Open Licence (open data)

The 12 SEARG soil-structure groups (``SEARG_Concise`` field)
-------------------------------------------------------------
    1.  Alluvial and coastal soils
    2.  Heavy clay soils with poor drainage
    3.  Light free drainage
    4.  Light soils with moderate & poor drainage
    5.  Man made
    6.  Medium soils with free drainage
    7.  Medium soils with moderate drainage
    8.  Medium with soils poor drainage
    9.  Organic soils with free drainage
    10. Organic soils with poor drainage
    11. Peat
    12. Shallow soils

The framework classifies Soil Associations from the National Soil Map (NATMAP)
into these groups according to inherent soil properties: texture, natural
drainage, flooding risk, soil depth, and organic matter content.

Typical usage
-------------
::

    from shapely.geometry import Point
    from eoflow.catchment import delineate_catchment
    from eoflow.soil import soil_coverage

    catchment = delineate_catchment(Point(-3.5, 50.7), "data/devon_dem.tif")
    coverage = soil_coverage(catchment)
    print(coverage)
    # SEARG_Concise
    # Light free drainage              0.412
    # Medium soils with free drainage  0.331
    # ...

Notes
-----
- Input polygons must be in **WGS 84 (EPSG:4326)**. The module reprojects
  internally to **British National Grid (EPSG:27700)** for the spatial query
  and area calculations (BNG is metric, so areas are in m²).
- The ArcGIS service returns at most ``maxRecordCount = 2000`` features per
  request; this module **paginates automatically**.
- All HTTP calls use ``requests``; supply a custom ``session`` for retries,
  proxies, or caching.
- Fractional coverage values are relative to the **total catchment area** in
  BNG.  They sum to ≤ 1.0; the gap (if any) represents catchment area with
  no SEARG classification.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from eoflow.log_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: ArcGIS Online item ID for the DEFRA Soil Structure Groups dataset.
ITEM_ID: str = "af498634e7c2409c8a0e3eecb720f8dc"

#: Root Feature Service URL.
FEATURE_SERVICE_URL: str = (
    "https://services1.arcgis.com/JZM7qJpmv7vJ0Hzx/arcgis/rest/services"
    "/SEARGFull_DEFRA12SoilGroups20240617/FeatureServer"
)

#: Layer-0 URL (the single polygon layer in this service).
LAYER_URL: str = f"{FEATURE_SERVICE_URL}/0"

#: Query endpoint for layer 0.
QUERY_URL: str = f"{LAYER_URL}/query"

#: Maximum records the service returns in a single response.
MAX_RECORD_COUNT: int = 2000

#: The 12 SEARG_Concise soil-structure group names, in alphabetical order.
SEARG_GROUPS: Tuple[str, ...] = (
    "Alluvial and coastal soils",
    "Heavy clay soils with poor drainage",
    "Light free drainage",
    "Light soils with moderate & poor drainage",
    "Man made",
    "Medium soils with free drainage",
    "Medium soils with moderate drainage",
    "Medium with soils poor drainage",
    "Organic soils with free drainage",
    "Organic soils with poor drainage",
    "Peat",
    "Shallow soils",
)

#: Official EA/DEFRA colour scheme for each SEARG_Concise group.
#: Matches the ArcGIS service renderer used in the EA online viewer.
SEARG_COLOURS: Dict[str, str] = {
    "Alluvial and coastal soils": "#5b9bd4",
    "Heavy clay soils with poor drainage": "#548034",
    "Light free drainage": "#ffc105",
    "Light soils with moderate & poor drainage": "#ffd966",
    "Man made": "#002673",
    "Medium soils with free drainage": "#dec98e",
    "Medium soils with moderate drainage": "#ab6a0f",
    "Medium with soils poor drainage": "#b09604",
    "Organic soils with free drainage": "#e6e6e6",
    "Organic soils with poor drainage": "#cfcfcf",
    "Peat": "#808080",
    "Shallow soils": "#ffff99",
}

#: Attribute fields requested from the service in every query.
_QUERY_FIELDS: Tuple[str, ...] = (
    "OBJECTID",
    "SEARG_Concise",
    "SEARG",
    "MU_NAME",
    "MAP_SYMBOL",
    "BFI",
    "SPR",
    "SEARGDescription",
    "Shape__Area",
)


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


def _to_bng(polygon: Polygon) -> Polygon:
    """Reproject a WGS-84 Shapely polygon to British National Grid (EPSG:27700).

    Parameters
    ----------
    polygon :
        Input polygon in WGS 84 (EPSG:4326).

    Returns
    -------
    shapely.geometry.Polygon
        The same polygon reprojected to EPSG:27700 (coordinates in metres).
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    return shapely_transform(transformer.transform, polygon)


def _polygon_to_arcgis_json(polygon) -> dict:
    """Encode a BNG Shapely Polygon or MultiPolygon as an ArcGIS REST geometry JSON object.

    ArcGIS represents all polygon geometries (including multi-part ones) as a
    single ``{"rings": [...], "spatialReference": {...}}`` object where each
    ring is a list of ``[x, y]`` coordinate pairs.  For a ``MultiPolygon`` we
    flatten the rings from every constituent part into one list.

    Parameters
    ----------
    polygon :
        Shapely ``Polygon`` or ``MultiPolygon`` in EPSG:27700 (BNG metres).

    Returns
    -------
    dict
        ArcGIS geometry JSON with ``"rings"`` and ``"spatialReference"``.
    """
    all_rings: list = []

    if polygon.geom_type == "MultiPolygon":
        for part in polygon.geoms:
            geom = part.__geo_interface__
            # GeoJSON "coordinates" for a Polygon: [[exterior], [hole1], ...]
            all_rings.extend([list(ring) for ring in geom["coordinates"]])
    else:
        geom = polygon.__geo_interface__
        all_rings = [list(ring) for ring in geom["coordinates"]]

    return {
        "rings": all_rings,
        "spatialReference": {"wkid": 27700},
    }


def _rings_to_shapely(rings: list) -> Polygon | MultiPolygon:
    """Convert an ArcGIS polygon rings list to a valid Shapely geometry.

    ArcGIS represents polygons as a list of coordinate rings where:
    - Outer rings have clockwise winding (but Shapely/GeoJSON use CCW for
      exteriors, so we let Shapely sort it out via ``buffer(0)``).
    - Holes have the opposite winding.
    - Multipart polygons have multiple outer rings.

    Parameters
    ----------
    rings :
        List of rings, each a list of ``[x, y]`` coordinate pairs.

    Returns
    -------
    shapely.geometry.Polygon or MultiPolygon
        A valid (or ``buffer(0)``-fixed) Shapely geometry.
    """
    if not rings:
        return Polygon()

    if len(rings) == 1:
        geom = Polygon(rings[0])
    else:
        # Attempt single polygon with holes (most common case)
        try:
            geom = Polygon(rings[0], rings[1:])
        except Exception:
            geom = Polygon()

        # If the result is invalid (multipart rather than holed polygon),
        # fall back to treating every ring as a separate polygon.
        if not geom.is_valid:
            parts = []
            for ring in rings:
                try:
                    parts.append(Polygon(ring))
                except Exception:
                    pass
            geom = unary_union(parts) if parts else Polygon()

    # Final validity fix
    if not geom.is_valid:
        geom = geom.buffer(0)

    return geom


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _paginated_query(
    geometry_json: dict,
    *,
    session: Optional[requests.Session],
    timeout: int,
) -> List[dict]:
    """Issue paginated POST requests to the SEARG query endpoint.

    Loops over pages using ``resultOffset`` until ``exceededTransferLimit``
    is ``False`` (or absent).

    Parameters
    ----------
    geometry_json :
        ArcGIS polygon geometry JSON (in BNG) for the spatial filter.
    session :
        Existing :class:`requests.Session`, or ``None`` to create one.
    timeout :
        HTTP timeout per request (seconds).

    Returns
    -------
    list of dict
        All feature dicts from all pages.

    Raises
    ------
    requests.HTTPError
        On a non-2xx HTTP response.
    RuntimeError
        If the ArcGIS service returns an ``"error"`` body.
    """
    sess = session or requests.Session()
    features: List[dict] = []
    offset = 0

    while True:
        params = {
            "geometry": json.dumps(geometry_json),
            "geometryType": "esriGeometryPolygon",
            "inSR": "27700",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": ",".join(_QUERY_FIELDS),
            "returnGeometry": "true",
            "outSR": "27700",
            "resultOffset": str(offset),
            "resultRecordCount": str(MAX_RECORD_COUNT),
            "f": "json",
        }

        resp = sess.post(QUERY_URL, data=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            err = data["error"]
            raise RuntimeError(
                f"ArcGIS service error {err.get('code', '?')}: "
                f"{err.get('message', 'unknown error')}"
            )

        batch = data.get("features", [])
        features.extend(batch)
        logger.debug(
            "SEARG query page offset=%d returned %d features (total so far: %d)",
            offset,
            len(batch),
            len(features),
        )

        if not data.get("exceededTransferLimit", False):
            break

        offset += MAX_RECORD_COUNT

    return features


def _features_to_geodataframe(features: List[dict]) -> gpd.GeoDataFrame:
    """Convert a list of ArcGIS feature dicts to a GeoDataFrame in EPSG:27700.

    Parameters
    ----------
    features :
        List of ``{"attributes": {...}, "geometry": {"rings": [...]}}`` dicts
        as returned by the ArcGIS REST query endpoint.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per feature, CRS EPSG:27700.  Returns an empty GeoDataFrame
        with the expected columns when *features* is empty.
    """
    if not features:
        return gpd.GeoDataFrame(
            columns=list(_QUERY_FIELDS) + ["geometry"],
            geometry="geometry",
            crs="EPSG:27700",
        )

    rows = []
    for feat in features:
        attrs = dict(feat.get("attributes", {}))
        geom_json = feat.get("geometry")

        if geom_json and "rings" in geom_json:
            geom = _rings_to_shapely(geom_json["rings"])
        else:
            geom = None

        rows.append({**attrs, "geometry": geom})

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:27700")
    return gdf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query_soil_polygons(
    polygon,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = 60,
) -> gpd.GeoDataFrame:
    """Query the SEARG feature service and return all soil polygons that
    intersect *polygon*.

    This is the low-level access function that returns the raw geometries and
    attributes from the service.  For the higher-level fractional coverage
    summary use :func:`soil_coverage` instead.

    Parameters
    ----------
    polygon :
        Area of interest in WGS 84 (EPSG:4326).  May be a
        ``shapely.geometry.Polygon`` or ``MultiPolygon`` (e.g. a county
        boundary).  Typically a catchment polygon returned by
        :func:`eoflow.catchment.delineate_catchment`.
    session :
        Optional :class:`requests.Session` for connection pooling, retries,
        or proxy settings.  A new session is created internally if not given.
    timeout :
        HTTP request timeout in seconds (default: 60).

    Returns
    -------
    geopandas.GeoDataFrame
        All SEARG soil polygons that spatially intersect *polygon*, in
        **British National Grid (EPSG:27700)**.  Key columns:

        - ``SEARG_Concise`` — one of the 12 SEARG group names
        - ``SEARG``         — the (sometimes longer) SEARG group label
        - ``MU_NAME``       — NATMAP mapping-unit name
        - ``MAP_SYMBOL``    — mapping-unit symbol code
        - ``BFI``           — Base Flow Index (float)
        - ``SPR``           — Standard Percentage Runoff (integer)
        - ``SEARGDescription`` — human-readable group description
        - ``Shape__Area``   — original polygon area from the service (m²)
        - ``geometry``      — Shapely Polygon/MultiPolygon in EPSG:27700

        Returns an **empty** GeoDataFrame (with the expected columns) when
        the polygon lies entirely outside the SEARG coverage area (England
        and Wales only).

    Raises
    ------
    requests.HTTPError
        On a non-2xx response from the service.
    RuntimeError
        If the ArcGIS service returns an error payload.
    ValueError
        If *polygon* cannot be reprojected (e.g. pyproj not installed).

    Examples
    --------
    ::

        from shapely.geometry import box
        from eoflow.soil import query_soil_polygons

        # Small bounding box over Devon
        devon_box = box(-3.6, 50.6, -3.4, 50.8)
        gdf = query_soil_polygons(devon_box)
        print(gdf[["MU_NAME", "SEARG_Concise", "Shape__Area"]].head())
    """
    polygon_bng = _to_bng(polygon)
    geom_json = _polygon_to_arcgis_json(polygon_bng)

    logger.info(
        "Querying SEARG service  bbox (BNG): %.0f,%.0f → %.0f,%.0f",
        *polygon_bng.bounds,
    )

    features = _paginated_query(geom_json, session=session, timeout=timeout)
    logger.info("Received %d SEARG soil polygon(s).", len(features))

    return _features_to_geodataframe(features)


def soil_coverage(
    polygon,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = 60,
    polygons_gdf: Optional[gpd.GeoDataFrame] = None,
) -> pd.Series:
    """Compute the fractional SEARG soil-group coverage within *polygon*.

    Each SEARG soil polygon returned by the service is **intersected** with
    *polygon* (both in BNG) and its clipped area is accumulated by
    ``SEARG_Concise`` group.  Results are expressed as fractions of the
    **total catchment area**.

    Parameters
    ----------
    polygon :
        Catchment polygon (``Polygon`` or ``MultiPolygon``) in WGS 84
        (EPSG:4326).
    session :
        Optional reusable :class:`requests.Session`.
    timeout :
        HTTP request timeout in seconds (default: 60).

    Returns
    -------
    pandas.Series
        Indexed by the 12 :data:`SEARG_GROUPS` names (alphabetical order),
        values are fractions **0.0–1.0** of the catchment area covered by
        each group.

        - Groups with no coverage return ``0.0``.
        - Values sum to **≤ 1.0**; the gap is the fraction of the catchment
          not covered by any SEARG polygon (e.g. if the catchment extends
          into Ireland or Scotland, or into the sea).
        - The series ``name`` is ``"soil_fraction"``.

    Raises
    ------
    ValueError
        If *polygon* has zero area after reprojection.
    requests.HTTPError
        On a non-2xx response from the service.
    RuntimeError
        If the ArcGIS service returns an error payload.

    Examples
    --------
    ::

        from shapely.geometry import box
        from eoflow.soil import soil_coverage

        devon_box = box(-3.6, 50.6, -3.4, 50.8)
        fractions = soil_coverage(devon_box)
        print(fractions[fractions > 0].sort_values(ascending=False))
        # Medium soils with free drainage    0.43
        # Light free drainage                0.29
        # ...
        print(f"Total classified: {fractions.sum():.1%}")
    """
    polygon_bng = _to_bng(polygon)
    catchment_area_m2 = polygon_bng.area

    if catchment_area_m2 == 0.0:
        raise ValueError("Input polygon has zero area after reprojection to BNG.")

    if polygons_gdf is not None:
        gdf = polygons_gdf
    else:
        gdf = query_soil_polygons(polygon, session=session, timeout=timeout)

    # Accumulate intersection area (m²) per SEARG_Concise group
    area_by_group: Dict[str, float] = {g: 0.0 for g in SEARG_GROUPS}

    if gdf.empty:
        logger.warning(
            "No SEARG polygons intersect the given catchment — returning all-zero coverage. "
            "This dataset covers England and Wales only."
        )
        return pd.Series(area_by_group, name="soil_fraction")

    unknown_groups: set = set()
    for _, row in gdf.iterrows():
        soil_geom = row.geometry
        group = row.get("SEARG_Concise")

        if soil_geom is None or soil_geom.is_empty:
            continue

        if group not in area_by_group:
            unknown_groups.add(str(group))
            continue

        try:
            intersection = polygon_bng.intersection(soil_geom)
            area_by_group[group] += intersection.area
        except Exception as exc:
            logger.warning(
                "Intersection failed for OBJECTID %s (%r): %s",
                row.get("OBJECTID"),
                group,
                exc,
            )

    if unknown_groups:
        logger.warning("Skipped %d unknown SEARG group(s): %s", len(unknown_groups), unknown_groups)

    fractions = {g: area / catchment_area_m2 for g, area in area_by_group.items()}
    series = pd.Series(fractions, name="soil_fraction")

    total = series.sum()
    logger.info(
        "Soil coverage complete: %.1f%% of catchment classified across %d group(s).",
        total * 100,
        (series > 0).sum(),
    )

    return series


def soil_coverage_summary(
    polygon,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Return a detailed summary DataFrame of soil coverage within *polygon*.

    Wraps :func:`soil_coverage` and adds the area in km² and a percentage
    column, sorted from largest to smallest coverage.  Groups with zero
    coverage are excluded.

    Parameters
    ----------
    polygon :
        Catchment polygon (``Polygon`` or ``MultiPolygon``) in WGS 84
        (EPSG:4326).
    session :
        Optional reusable :class:`requests.Session`.
    timeout :
        HTTP request timeout in seconds.

    Returns
    -------
    pandas.DataFrame
        Columns: ``SEARG_Concise``, ``fraction``, ``percent``,
        ``area_km2``.  Sorted descending by ``fraction``.  Empty rows
        (fraction == 0) are dropped.

    Examples
    --------
    ::

        from shapely.geometry import box
        from eoflow.soil import soil_coverage_summary

        devon_box = box(-3.6, 50.6, -3.4, 50.8)
        df = soil_coverage_summary(devon_box)
        print(df.to_string(index=False))
    """
    polygon_bng = _to_bng(polygon)
    catchment_area_m2 = polygon_bng.area
    catchment_area_km2 = catchment_area_m2 / 1_000_000.0

    fractions = soil_coverage(polygon, session=session, timeout=timeout)

    df = pd.DataFrame(
        {
            "SEARG_Concise": fractions.index,
            "fraction": fractions.values,
            "percent": fractions.values * 100.0,
            "area_km2": fractions.values * catchment_area_km2,
        }
    )

    df = df[df["fraction"] > 0].sort_values("fraction", ascending=False).reset_index(drop=True)
    return df
