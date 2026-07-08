"""
Visualise EA water-quality sample points on an interactive map.

Reads a CSV produced by ``convert_ea_csv.py`` (or the raw EA CSV) and
displays the sampling locations as colour-coded circle markers on a
`leafmap <https://leafmap.org>`__ map.  The resulting HTML page is
written to a temporary file and opened automatically in the default
web browser.

Usage
-----
    python -m scripts.visualise_ea_samples \
        --input ea_water_quality_clean.csv

    # Optionally specify which value column to colour by
    python -m scripts.visualise_ea_samples \
        --input ea_water_quality_clean.csv \
        --value-column "Turbidity (NEPHELOMETRIC TURBIDITY UNITS)"
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import webbrowser
from pathlib import Path

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd

from eoflow.log_utils import get_logger
from eoflow.utils import easting_northing_to_latlon

logger = get_logger(__file__)

# ---- Well-known column sets ------------------------------------------------

_META_COLS = {
    "id",
    "samplingPoint.notation",
    "samplingPoint.prefLabel",
    "samplingPoint.easting",
    "samplingPoint.northing",
    "samplingPoint.region",
    "samplingPoint.area",
    "samplingPoint.subArea",
    "samplingPoint.samplingPointStatus",
    "samplingPoint.samplingPointType",
    "phenomenonTime",
    "samplingPurpose",
    "sampleMaterialType",
    "determinand.prefLabel",
    "determinand.notation",
    "unit",
    "result",
    "latitude",
    "longitude",
}


# ---- Helpers ---------------------------------------------------------------


def _ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``latitude`` / ``longitude`` columns if they are missing.

    Falls back to converting ``samplingPoint.easting`` /
    ``samplingPoint.northing`` via :func:`easting_northing_to_latlon`.
    """
    if "latitude" in df.columns and "longitude" in df.columns:
        return df

    if "samplingPoint.easting" not in df.columns or "samplingPoint.northing" not in df.columns:
        raise ValueError(
            "CSV must contain either (latitude, longitude) or "
            "(samplingPoint.easting, samplingPoint.northing) columns."
        )

    eastings = np.asarray(df["samplingPoint.easting"].values, dtype=float)
    northings = np.asarray(df["samplingPoint.northing"].values, dtype=float)
    lat, lon = easting_northing_to_latlon(eastings, northings)
    df = df.copy()
    df["latitude"] = lat
    df["longitude"] = lon
    return df


def _detect_value_column(df: pd.DataFrame) -> str:
    """Heuristically pick the measurement-value column.

    If the CSV still has a ``result`` column (raw format) that is used.
    Otherwise, the first column whose name is *not* in the known metadata
    set is returned.
    """
    if "result" in df.columns:
        return "result"

    candidates = [c for c in df.columns if c not in _META_COLS]
    if not candidates:
        raise ValueError(
            "Could not auto-detect a value column.  Use --value-column to specify one explicitly."
        )
    return candidates[0]


def _build_popup_html(row: pd.Series, value_col: str) -> str:
    """Return a small HTML snippet used inside a Folium popup."""
    name = row.get("samplingPoint.prefLabel", "")
    notation = row.get("samplingPoint.notation", "")
    region = row.get("samplingPoint.region", "")
    area = row.get("samplingPoint.area", "")
    time = row.get("phenomenonTime", "")
    value = row.get(value_col, "")
    lat = row.get("latitude", "")
    lon = row.get("longitude", "")

    lines = [
        f"<b>{name}</b>",
        f"Notation: {notation}",
        f"Region: {region}",
        f"Area: {area}",
        f"Time: {time}",
        f"<b>{value_col}: {value}</b>",
        f"Lat: {lat:.5f}, Lon: {lon:.5f}" if isinstance(lat, float) else "",
    ]
    return "<br>".join(line for line in lines if line)


# ---- Core visualisation ----------------------------------------------------


def visualise(
    input_path: Path,
    value_column: str | None = None,
    output_html: Path | None = None,
) -> Path:
    """Build an interactive map and open it in the browser.

    Parameters
    ----------
    input_path : Path
        CSV file to read (clean or raw EA format).
    value_column : str or None
        Column whose values drive the marker colour.  Auto-detected when
        *None*.
    output_html : Path or None
        Where to write the HTML file.  A temporary file is used when *None*.

    Returns
    -------
    Path
        The path to the written HTML file.
    """
    df = pd.read_csv(input_path)
    df = _ensure_latlon(df)

    # --- Determine the value column & coerce to numeric -------------------
    value_col = value_column or _detect_value_column(df)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in CSV. Available: {list(df.columns)}")

    df["_value"] = pd.to_numeric(df[value_col], errors="coerce")

    # Drop rows that have no location or no value
    plot_df = df.dropna(subset=["latitude", "longitude", "_value"]).copy()
    if plot_df.empty:
        raise ValueError("No plottable rows remain after dropping NaN lat/lon/values.")

    # --- Colour scale (log-ish for right-skewed EA data) ------------------
    vmin = float(plot_df["_value"].min())
    vmax = float(plot_df["_value"].max())

    # Use a log scale when the range spans more than one order of magnitude
    # and all values are positive – this gives much better visual contrast
    # for typically right-skewed water-quality measurements.
    use_log = vmin > 0 and vmax / vmin > 10

    if use_log:
        log_min = math.log10(vmin)
        log_max = math.log10(vmax)

        def _norm(v: float) -> float:
            if log_max == log_min:
                return 0.5
            return (math.log10(v) - log_min) / (log_max - log_min)

        colormap = cm.LinearColormap(
            colors=["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"],
            vmin=vmin,
            vmax=vmax,
            caption=f"{value_col} (log scale)",
        )
    else:

        def _norm(v: float) -> float:
            if vmax == vmin:
                return 0.5
            return (v - vmin) / (vmax - vmin)

        colormap = cm.LinearColormap(
            colors=["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"],
            vmin=vmin,
            vmax=vmax,
            caption=value_col,
        )

    # Precompute a hex colour for each row
    plot_df["_colour"] = plot_df["_value"].apply(lambda v: colormap(v))

    # --- Build the Folium map ---------------------------------------------
    centre_lat = float(plot_df["latitude"].mean())
    centre_lon = float(plot_df["longitude"].mean())

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=6,
        tiles="OpenStreetMap",
    )

    # Markers
    for _, row in plot_df.iterrows():
        val = float(row["_value"])
        radius = 4 + 8 * _norm(val)  # 4–12 px
        popup_html = _build_popup_html(row, value_col)

        lat = float(row["latitude"])
        lon = float(row["longitude"])
        colour = str(row["_colour"])

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.8,
            weight=1,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{row.get('samplingPoint.prefLabel', '')}: {val}",
        ).add_to(m)

    # Add the colour legend
    colormap.add_to(m)

    # Fit the map to the data bounds
    sw = [float(plot_df["latitude"].min()), float(plot_df["longitude"].min())]
    ne = [float(plot_df["latitude"].max()), float(plot_df["longitude"].max())]
    m.fit_bounds([sw, ne], padding=[30, 30])

    # --- Write HTML and open in browser -----------------------------------
    if output_html is not None:
        html_path = Path(output_html).resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".html", prefix="ea_samples_map_", delete=False)
        tmp.close()
        html_path = Path(tmp.name).resolve()

    m.save(str(html_path))
    logger.info("Map saved to %s  (%d points)", html_path, len(plot_df))
    webbrowser.open(html_path.as_uri())
    return html_path


# ---- CLI -------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualise EA water-quality samples as colour-coded points "
            "on an interactive map opened in the browser."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("ea_water_quality_clean.csv"),
        help="Path to the (clean or raw) EA CSV file.",
    )
    p.add_argument(
        "--value-column",
        type=str,
        default=None,
        help=("Name of the column to colour markers by.  Auto-detected when omitted."),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Path to write the HTML map.  A temporary file is used when omitted."),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    try:
        visualise(args.input, value_column=args.value_column, output_html=args.output)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
