"""
Print a quick summary report for a water-quality samples CSV, and plot violin
charts of each site's sample-date distribution — one across the full date
range, one sliced to a single calendar year.

Usage
-----
    python -m scripts.gpkg_report <input.gpkg> [plot_output.png] [year]

*year* defaults to the most recent year present in the data.
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eoflow.log_utils import get_logger

logger = get_logger(__file__)

#: Single-hue fill for the violins (dataviz skill default sequential blue).
_VIOLIN_COLOR = "#2a78d6"
_VIOLIN_EDGE = "#52514e"
_MEDIAN_COLOR = "#0b0b0b"
_POINT_COLOR = "#0b0b0b"


def _grouped_sample_dates(
    gdf: gpd.GeoDataFrame,
    site_col: str,
    date_col: str,
    *,
    year: int | None = None,
) -> "pd.Series":
    """Per-site arrays of ``date2num`` sample dates, sorted by each site's mean date.

    When *year* is given, only samples from that calendar year are kept.
    """
    dates = pd.to_datetime(gdf[date_col], errors="coerce")
    has_site_date = dates.notna() & gdf[site_col].notna()
    n_dropped = (~has_site_date).sum()
    if n_dropped:
        logger.warning("Dropping %d row(s) with missing site/date for the frequency plot", n_dropped)

    valid = has_site_date if year is None else has_site_date & (dates.dt.year == year)
    df = pd.DataFrame({"site": gdf.loc[valid, site_col], "date": dates[valid]})
    if df.empty:
        raise ValueError("No samples with a valid site and date to plot" + (f" for year {year}" if year else ""))

    by_site = df.groupby("site")["date"].apply(lambda s: mdates.date2num(s.sort_values()))
    return by_site.loc[by_site.apply(lambda a: a.mean()).sort_values().index]


def _draw_violins_with_points(
    ax: "plt.Axes",
    by_site: "pd.Series",
    *,
    violin_width: float,
    jitter_seed: int,
) -> range:
    """Draw jittered individual-sample points + violins for each site onto *ax*.

    Returns the x positions used (1..n_sites), for the caller to set tick labels.
    """
    rng = np.random.default_rng(jitter_seed)
    jitter_half_span = violin_width * 0.45

    positions = range(1, len(by_site) + 1)
    violin_data, violin_positions = [], []
    for pos, values in zip(positions, by_site):
        jitter = rng.uniform(-jitter_half_span, jitter_half_span, size=len(values))
        ax.scatter(
            pos + jitter,
            values,
            color=_POINT_COLOR,
            s=8,
            alpha=0.5,
            linewidths=0,
            zorder=4,
        )
        if len(set(values)) >= 2:
            violin_data.append(values)
            violin_positions.append(pos)

    if violin_data:
        parts = ax.violinplot(
            violin_data,
            positions=violin_positions,
            widths=violin_width,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(_VIOLIN_COLOR)
            body.set_edgecolor(_VIOLIN_EDGE)
            body.set_alpha(0.6)
        parts["cmedians"].set_color(_MEDIAN_COLOR)

    return positions


def plot_sample_frequency_by_site(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    *,
    site_col: str = "samplingPoint.notation",
    date_col: str = "phenomenonTime",
    violin_width: float = 0.8,
    jitter_seed: int = 0,
) -> Path:
    """Violin plot of each site's sample-date distribution over time.

    One violin per site, ordered left-to-right by mean sample date, shows
    how that site's visits are distributed in time — e.g. dense/regular
    monitoring vs. a handful of one-off samples. Every individual sample
    date is also drawn as a small jittered point on top of (or in place of,
    for sites with fewer than two distinct dates) its violin, so exact
    sample times remain visible rather than only the smoothed shape.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain *site_col* and *date_col*.
    output_path : Path
        Destination PNG file.
    site_col : str
        Column identifying each site.
    date_col : str
        Column holding the sample timestamp (parseable by
        :func:`pandas.to_datetime`).
    violin_width : float
        Width of each violin/jitter band, in the same units as the spacing
        between adjacent site columns (1.0). Sites are spaced 1.0 apart, so
        values approaching or exceeding 1.0 will make neighbouring sites'
        violins/points visually overlap; the default 0.8 leaves a small gap.
    jitter_seed : int
        Seed for the horizontal jitter applied to individual sample points,
        so the same input produces the same plot.

    Returns
    -------
    Path
        *output_path* after writing.
    """
    by_site = _grouped_sample_dates(gdf, site_col, date_col)

    n_sites = len(by_site)
    logger.info(
        "Plotting %d sites; violin_width=%.2f (site columns are spaced 1.0 apart, "
        "so this is also the max span before adjacent sites' violins/points overlap)",
        n_sites,
        violin_width,
    )
    fig, ax = plt.subplots(figsize=(max(14, n_sites * 0.28), 8))
    positions = _draw_violins_with_points(ax, by_site, violin_width=violin_width, jitter_seed=jitter_seed)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(by_site.index), rotation=90, fontsize=6)
    ax.yaxis.set_major_locator(mdates.YearLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Site")
    ax.set_ylabel("Sample date")
    ax.set_title(f"Sample-date distribution by site (n={n_sites} sites, violin_width={violin_width:.2f})")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sample-frequency violin plot saved to %s", output_path)
    return output_path


def plot_sample_frequency_by_site_for_year(
    gdf: gpd.GeoDataFrame,
    year: int,
    output_path: Path,
    *,
    site_col: str = "samplingPoint.notation",
    date_col: str = "phenomenonTime",
    violin_width: float = 0.8,
    jitter_seed: int = 0,
) -> Path:
    """Same as :func:`plot_sample_frequency_by_site`, sliced to one calendar year.

    Samples outside *year* are dropped before plotting, and the y-axis
    becomes month-within-year rather than year, so this shows each site's
    seasonal sampling pattern for *year* specifically.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain *site_col* and *date_col*.
    year : int
        Calendar year to restrict samples to.
    output_path : Path
        Destination PNG file.
    site_col, date_col, violin_width, jitter_seed
        See :func:`plot_sample_frequency_by_site`.

    Returns
    -------
    Path
        *output_path* after writing.

    Raises
    ------
    ValueError
        If no samples fall within *year*.
    """
    by_site = _grouped_sample_dates(gdf, site_col, date_col, year=year)

    n_sites = len(by_site)
    logger.info(
        "Plotting %d sites for year %d; violin_width=%.2f (site columns are spaced 1.0 apart, "
        "so this is also the max span before adjacent sites' violins/points overlap)",
        n_sites,
        year,
        violin_width,
    )
    fig, ax = plt.subplots(figsize=(max(14, n_sites * 0.28), 8))
    positions = _draw_violins_with_points(ax, by_site, violin_width=violin_width, jitter_seed=jitter_seed)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(by_site.index), rotation=90, fontsize=6)
    ax.set_ylim(mdates.date2num(pd.Timestamp(year, 1, 1)), mdates.date2num(pd.Timestamp(year, 12, 31)))
    ax.yaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("Site")
    ax.set_ylabel(f"Sample date ({year})")
    ax.set_title(f"Sample-date distribution by site — {year} (n={n_sites} sites, violin_width={violin_width:.2f})")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sample-frequency violin plot (year=%d) saved to %s", year, output_path)
    return output_path


def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python -m scripts.gpkg_report <input.gpkg> [plot_output.png] [year]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.is_file():
        logger.error("Input file not found: %s", input_file)
        sys.exit(1)
    if input_file.suffix != ".gpkg" and input_file.suffix != ".pkg":
        logger.error("Input file must be a GeoPackage file: %s", input_file)
        sys.exit(1)

    plot_output = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file.with_name(f"{input_file.stem}_sample_frequency.png")
    year_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None

    logger.info("Reporting on %s", input_file)
    gdf = gpd.read_file(input_file)

    n_polygons = len(gdf["geometry"].unique())
    n_locations = len(gdf.groupby(["latitude", "longitude"]))

    # The report itself is the script's intended stdout output, so it is
    # printed directly rather than routed through the logger.
    print(f"Reporting on {input_file}")
    print(f"Total samples: {len(gdf)}")
    print(f"Unique sampling locations: {len(gdf.groupby(['latitude', 'longitude']))}")
    print(f"Date coverage: {gdf['phenomenonTime'].min()} to {gdf['phenomenonTime'].max()}")
    print(f"Lat coverage: {gdf['latitude'].min()} to {gdf['latitude'].max()}")
    print(f"Lon coverage: {gdf['longitude'].min()} to {gdf['longitude'].max()}")
    print(f"Catchments delineated: {n_polygons}/{n_locations}")
    print(
        f"Samples delineated: {gdf['geometry'].notnull().sum()}/{gdf.shape[0]} ({gdf['geometry'].notnull().sum() / gdf.shape[0] * 100:.2f}%)"
    )
    print("")
    print("GDF head:")
    print(f"{gdf.head(5)}")
    print("GDF tail:")
    print(f"{gdf.tail(5)}")

    plot_sample_frequency_by_site(gdf, plot_output)

    year = year_arg if year_arg is not None else int(pd.to_datetime(gdf["phenomenonTime"], errors="coerce").dt.year.max())
    year_output = plot_output.with_name(f"{plot_output.stem}_{year}.png")
    plot_sample_frequency_by_site_for_year(gdf, year, year_output)


if __name__ == "__main__":
    main()
