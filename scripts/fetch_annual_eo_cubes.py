"""
Fetch whole-region, per-year Sentinel-2 band/index datacubes covering every
delineated catchment in a :class:`~eoflow.samples.CatchmentDataset`
GeoPackage (as produced by ``delineate_catchments.py``).

Rationale
---------
Querying EO data separately for each sample's small catchment polygon means
re-downloading heavily overlapping regions many times over.  This script
instead fetches one datacube per year, covering the union bounding box of
*all* catchments, and lets :mod:`scripts.build_eo_dataset` index into that
single per-year source using each sample's own catchment polygon.

Because a whole-county, full-year request is far too large for openEO's
synchronous download API (see ``scripts.get_eo_data_by_shapefile``), this
script submits each request as an openEO **batch job**
(:func:`eoflow.eo.download_cube_as_batch_job`), which blocks until the job
completes but has no such size restriction.

Each band group and each index is saved as its own NetCDF file, one
directory per year::

    <output-dir>/2024/bands.nc
    <output-dir>/2024/ndvi.nc
    <output-dir>/2024/ndwi.nc
    <output-dir>/2025/bands.nc
    ...

Bands, indices, cloud-cover threshold, collection, backend, and bbox padding
all default to the ``eo`` section of :mod:`eoflow.config` and can be
overridden per-invocation via CLI flags.

Usage
-----
    python -m scripts.fetch_annual_eo_cubes \\
        --gpkg data/delineated_catchments/devon_turbidity_sites_2010-10_2026-06.pkg \\
        --output-dir data/eo_cubes/devon_annual

    # Override bands/indices/window for this run only
    python -m scripts.fetch_annual_eo_cubes \\
        --gpkg data/delineated_catchments/devon_turbidity_sites_2010-10_2026-06.pkg \\
        --output-dir data/eo_cubes/devon_annual \\
        --bands B02 B03 B04 B08 \\
        --indices NDVI NDWI \\
        --start-year 2020 --end-year 2024

Notes
-----
* Requires the optional ``openeo`` dependency (see :mod:`eoflow.eo`).
* A whole-county, 10 m-resolution, full-year Sentinel-2 cube is a genuinely
  large request — expect batch jobs to take a long time and consume
  significant backend processing budget/storage. Narrow ``--start-year``/
  ``--end-year`` or the catchment set first if you just want to try this out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from eoflow import eo
from eoflow.config import get_config
from eoflow.log_utils import get_logger
from eoflow.samples import CatchmentDataset

logger = get_logger(__file__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _union_bbox(dataset: CatchmentDataset, pad: float) -> dict:
    """Union bounding box (WGS 84) of every catchment polygon in *dataset*."""
    gdf = dataset.with_catchments().to_geodataframe()
    gdf = gdf[gdf.geometry.notna()]
    if gdf.empty:
        raise ValueError("No delineated catchments found in the dataset.")

    west, south, east, north = gdf.total_bounds
    return {
        "west": float(west - pad),
        "south": float(south - pad),
        "east": float(east + pad),
        "north": float(north + pad),
    }


def _year_range(
    dataset: CatchmentDataset,
    start_year: Optional[int],
    end_year: Optional[int],
) -> range:
    """Resolve the (inclusive) year range to fetch.

    Explicit *start_year*/*end_year* take precedence; any value left as
    *None* falls back to the min/max year found across sample dates.
    """
    if start_year is not None and end_year is not None:
        return range(start_year, end_year + 1)

    years = [s.date.year for s in dataset if s.date is not None]
    if not years:
        raise ValueError(
            "No sample dates found in the dataset to infer a year range from; "
            "specify --start-year and --end-year explicitly."
        )
    return range(start_year if start_year is not None else min(years), (end_year if end_year is not None else max(years)) + 1)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def fetch_annual_cubes(
    gpkg_path: Path,
    output_dir: Path,
    *,
    bands: Optional[Sequence[str]] = None,
    indices: Optional[Sequence[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    pad: Optional[float] = None,
    collection: Optional[str] = None,
    max_cloud_cover: Optional[int] = None,
    backend: Optional[str] = None,
    authenticate: bool = True,
) -> List[Path]:
    """Fetch one Sentinel-2 band/index datacube per year for the union of catchments.

    Parameters
    ----------
    gpkg_path : Path
        GeoPackage produced by ``delineate_catchments.py``.
    output_dir : Path
        Root directory to write ``<year>/bands.nc`` / ``<year>/{index}.nc``
        files into.
    bands : sequence of str, optional
        Band names to download together as a single multi-band datacube per
        year. Defaults to the ``eo.bands`` config value.
    indices : sequence of str, optional
        Spectral index names to download, each saved as its own datacube per
        year. Defaults to the ``eo.indices`` config value.
    start_year, end_year : int, optional
        Inclusive year range to fetch. Defaults to the min/max year found
        across sample dates in the GeoPackage.
    pad : float, optional
        Degrees of padding added around the union catchment bounding box.
        Defaults to the ``eo.bbox_pad`` config value.
    collection : str, optional
        openEO collection ID. Defaults to the ``eo.collection`` config value.
    max_cloud_cover : int, optional
        Maximum cloud cover percentage (0-100). Defaults to the
        ``eo.max_cloud_cover`` config value.
    backend : str, optional
        openEO backend URL. Defaults to the ``eo.backend`` config value.
    authenticate : bool
        Whether to authenticate the openEO connection.

    Returns
    -------
    list of Path
        Paths to all NetCDF files successfully written.
    """
    config = get_config()
    bands = list(bands) if bands is not None else list(config.get("eo.bands", []))
    indices = list(indices) if indices is not None else list(config.get("eo.indices", []))
    pad = pad if pad is not None else config.get("eo.bbox_pad", 0.01)
    collection = collection if collection is not None else config.get("eo.collection", eo.SENTINEL2_COLLECTION)
    max_cloud_cover = max_cloud_cover if max_cloud_cover is not None else config.get("eo.max_cloud_cover", 85)
    backend = backend if backend is not None else config.get("eo.backend", eo.OPENEO_BACKEND)

    if not bands and not indices:
        raise ValueError(
            "Nothing to fetch: specify --bands and/or --indices (or set them "
            "under the 'eo' section of the eoflow config)."
        )

    logger.info("Loading catchment dataset from %s …", gpkg_path)
    dataset = CatchmentDataset.from_gpkg(gpkg_path, only_delineated=True)
    logger.info("Loaded %d delineated sample(s)", len(dataset))

    spatial_extent = _union_bbox(dataset, pad)
    logger.info(
        "Union catchment bbox (pad=%.4f°): west=%.5f south=%.5f east=%.5f north=%.5f",
        pad,
        spatial_extent["west"],
        spatial_extent["south"],
        spatial_extent["east"],
        spatial_extent["north"],
    )

    years = _year_range(dataset, start_year, end_year)
    logger.info("Fetching years: %s", list(years))

    logger.info("Connecting to openEO backend %s …", backend)
    connection = eo.connect(backend, authenticate=authenticate)

    output_dir = Path(output_dir)
    written: List[Path] = []

    for year in years:
        year_dir = output_dir / str(year)
        temporal_extent = (f"{year}-01-01", f"{year + 1}-01-01")

        if bands:
            band_path = year_dir / "bands.nc"
            try:
                cube = eo.build_band_cube(
                    connection,
                    spatial_extent,
                    temporal_extent,
                    bands,
                    collection=collection,
                    max_cloud_cover=max_cloud_cover,
                )
                eo.download_cube_as_batch_job(cube, band_path, title=f"eoflow bands {year}")
                written.append(band_path)
            except Exception as exc:
                logger.error("Failed to fetch bands %s for %d: %s", bands, year, exc)

        for index in indices:
            index_path = year_dir / f"{index.lower()}.nc"
            try:
                cube = eo.build_index_cube(
                    connection,
                    index,
                    spatial_extent,
                    temporal_extent,
                    collection=collection,
                    max_cloud_cover=max_cloud_cover,
                )
                eo.download_cube_as_batch_job(cube, index_path, title=f"eoflow {index.upper()} {year}")
                written.append(index_path)
            except Exception as exc:
                logger.error("Failed to fetch %s for %d: %s", index.upper(), year, exc)

    n_requested = ((1 if bands else 0) + len(indices)) * len(list(years))
    logger.info("Done: %d/%d datacube(s) written to %s", len(written), n_requested, output_dir)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config = get_config()
    p = argparse.ArgumentParser(
        description=(
            "Fetch whole-region, per-year Sentinel-2 band/index datacubes "
            "covering every delineated catchment in a GeoPackage, via openEO "
            "batch jobs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gpkg",
        type=Path,
        required=True,
        help="Path to the delineated-catchments GeoPackage (from delineate_catchments.py).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory to write <year>/bands.nc and <year>/{index}.nc into.",
    )
    p.add_argument(
        "--bands",
        nargs="+",
        default=None,
        metavar="BAND",
        help=f"Band names to fetch together per year. Defaults to config eo.bands ({config.get('eo.bands')}).",
    )
    p.add_argument(
        "--indices",
        nargs="+",
        default=None,
        metavar="INDEX",
        type=str.upper,
        choices=sorted(eo.SUPPORTED_INDICES),
        help=(
            "Spectral index names to fetch, each saved as its own datacube "
            f"per year. Defaults to config eo.indices ({config.get('eo.indices')}). "
            f"Supported: {', '.join(sorted(eo.SUPPORTED_INDICES))}."
        ),
    )
    p.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First year to fetch (inclusive). Defaults to the earliest sample date's year.",
    )
    p.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last year to fetch (inclusive). Defaults to the latest sample date's year.",
    )
    p.add_argument(
        "--pad",
        type=float,
        default=None,
        help=f"Degrees of padding around the union catchment bbox. Defaults to config eo.bbox_pad ({config.get('eo.bbox_pad')}).",
    )
    p.add_argument(
        "--collection",
        default=None,
        help=f"openEO collection ID. Defaults to config eo.collection ({config.get('eo.collection')}).",
    )
    p.add_argument(
        "--max-cloud-cover",
        type=int,
        default=None,
        help=f"Maximum cloud cover percentage (0-100). Defaults to config eo.max_cloud_cover ({config.get('eo.max_cloud_cover')}).",
    )
    p.add_argument(
        "--backend",
        default=None,
        help=f"openEO backend URL. Defaults to config eo.backend ({config.get('eo.backend')}).",
    )
    p.add_argument(
        "--no-auth",
        dest="authenticate",
        action="store_false",
        default=True,
        help="Skip OIDC authentication (only publicly accessible collections will work).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.gpkg.exists():
        logger.error("GeoPackage not found: %s", args.gpkg)
        sys.exit(1)

    try:
        fetch_annual_cubes(
            gpkg_path=args.gpkg,
            output_dir=args.output_dir,
            bands=args.bands,
            indices=args.indices,
            start_year=args.start_year,
            end_year=args.end_year,
            pad=args.pad,
            collection=args.collection,
            max_cloud_cover=args.max_cloud_cover,
            backend=args.backend,
            authenticate=args.authenticate,
        )
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
