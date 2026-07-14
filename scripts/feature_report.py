"""
Print a quick summary report for an extracted-features CSV (from
extract_features.py), and plot a data-completeness bar chart by
feature group.

Usage
-----
    python -m scripts.feature_report <features.csv> [plot_output.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eoflow.features import print_feature_summary, summarize_features
from eoflow.log_utils import get_logger

logger = get_logger(__file__)

#: Same palette as gpkg_report.py's violins, for visual consistency across report scripts.
_BAR_COLOR = "#2a78d6"
_BAR_EDGE = "#52514e"


def plot_group_completeness(summary: pd.DataFrame, output_path: Path) -> Path:
    """Horizontal bar chart of mean non-NaN fraction per feature group.

    Parameters
    ----------
    summary : pd.DataFrame
        As returned by :func:`eoflow.features.summarize_features`.
    output_path : Path
        Destination PNG file.

    Returns
    -------
    Path
        *output_path* after writing.
    """
    groups = summary.index.tolist()
    values = (summary["completeness"] * 100).fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(8, max(3, len(groups) * 0.5)))
    ax.barh(groups, values, color=_BAR_COLOR, edgecolor=_BAR_EDGE, zorder=3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Non-NaN cells (%)")
    ax.set_title("Feature completeness by group")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for y, v in enumerate(values):
        ax.text(min(v + 1, 96), y, f"{v:.0f}%", va="center", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature-completeness plot saved to %s", output_path)
    return output_path


def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python -m scripts.feature_report <input.csv> [plot_output.png]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.is_file():
        logger.error("Input file not found: %s", input_file)
        sys.exit(1)
    if input_file.suffix != ".csv":
        logger.error("Input file must be a CSV file: %s", input_file)
        sys.exit(1)

    plot_output = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else input_file.with_name(f"{input_file.stem}_completeness.png")
    )

    logger.info("Reporting on %s", input_file)
    df = pd.read_csv(input_file)

    # The report itself is the script's intended stdout output, so it is
    # printed directly rather than routed through the logger.
    print(f"Reporting on {input_file}")
    print(f"Total samples: {len(df)}")
    if "notation" in df.columns:
        print(f"Unique sites: {df['notation'].nunique()}")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        print(f"Date coverage: {dates.min()} to {dates.max()}")
    if "result" in df.columns:
        result = pd.to_numeric(df["result"], errors="coerce")
        print(
            f"Target (result) range: {result.min():.3g} to {result.max():.3g} "
            f"(mean {result.mean():.3g})"
        )

    print_feature_summary(df, input_file)

    summary = summarize_features(df)
    plot_group_completeness(summary, plot_output)


if __name__ == "__main__":
    main()
