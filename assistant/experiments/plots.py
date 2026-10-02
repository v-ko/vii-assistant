"""Generate presentable Plotly charts from experiment results.

Produces static PNG images (via kaleido) suitable for pasting into Google Docs,
plus a self-contained interactive HTML for exploration.

Usage:
    python -m assistant.experiments.plots <results_dir> [<results_dir2> ...]

For a single directory, charts are written into ``<results_dir>/plots/``.
When several directories are given, an extra cross-run comparison chart is
written into the first directory's ``plots/`` folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

from assistant.experiments.stats import load_results, load_run_params

# Image export settings — 2x scale gives crisp text when embedded in docs.
_IMG_WIDTH = 900
_IMG_HEIGHT = 500
_IMG_SCALE = 2

# Consistent, doc-friendly palette.
_PRIMARY = "#2563eb"  # blue
_PALETTE = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#dc2626", "#0891b2"]

# IoU histogram bins: full [0, 1] range, fixed width so every run is binned
# identically and is therefore directly comparable.
_BIN_SIZE = 0.05  # -> 20 bins across [0, 1]
# Headroom factor applied to the tallest bar when fixing the shared y-axis.
_Y_HEADROOM = 1.08


def _base_layout(title: str) -> dict:
    """Shared layout for a clean, presentation-ready look."""
    return dict(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)),
        template="plotly_white",
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=14),
        margin=dict(l=70, r=40, t=70, b=60),
        width=_IMG_WIDTH,
        height=_IMG_HEIGHT,
    )


def _ious(results: list[dict]) -> list[float]:
    """IoU values, clamped to the [0, 1] histogram range."""
    return [min(max(r["iou"], 0.0), 1.0) for r in results]


def _hist_peak_percent(results: list[dict]) -> float:
    """Tallest bar (as a percentage of samples) for the fixed-bin histogram."""
    ious = _ious(results)
    total = len(ious) or 1
    n_bins = round(1.0 / _BIN_SIZE)
    counts = [0] * n_bins
    for v in ious:
        idx = min(int(v / _BIN_SIZE), n_bins - 1)
        counts[idx] += 1
    return 100.0 * max(counts) / total


def _save(fig: go.Figure, out_dir: Path, name: str) -> Path:
    """Write a figure as a PNG and return its path."""
    path = out_dir / f"{name}.png"
    fig.write_image(str(path), scale=_IMG_SCALE)
    return path


def _run_label(results_dir: Path) -> str:
    """Human-friendly label for a run: model name if known, else dir name."""
    params = load_run_params(results_dir)
    if params and params.get("model"):
        return str(params["model"])
    return results_dir.name


def plot_iou_histogram(
    results: list[dict], title: str, y_max: float | None = None
) -> go.Figure:
    """IoU distribution as a fixed-bin histogram (percent of samples).

    The x-axis spans the full [0, 1] range with identical bins for every run,
    and the y-axis is in percent. Pass ``y_max`` to fix the y-axis to a shared
    scale so multiple histograms are visually comparable.
    """
    fig = go.Figure(
        go.Histogram(
            x=_ious(results),
            xbins=dict(start=0.0, end=1.0, size=_BIN_SIZE),
            histnorm="percent",
            marker_color=_PRIMARY,
            marker_line=dict(color="white", width=1),
            hovertemplate="IoU %{x}<br>%{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(f"IoU distribution — {title}"))
    fig.update_xaxes(title="IoU", range=[0, 1], dtick=0.1)
    fig.update_yaxes(title="% of samples", range=[0, y_max] if y_max else None)
    return fig


def plot_run_comparison(
    runs: list[tuple[str, list[dict]]], y_max: float | None = None
) -> go.Figure:
    """Overlaid IoU histograms comparing several runs on a shared scale."""
    fig = go.Figure()
    for idx, (label, results) in enumerate(runs):
        fig.add_trace(
            go.Histogram(
                name=label,
                x=_ious(results),
                xbins=dict(start=0.0, end=1.0, size=_BIN_SIZE),
                histnorm="percent",
                marker_color=_PALETTE[idx % len(_PALETTE)],
                opacity=0.6,
                hovertemplate=f"{label}<br>IoU %{{x}}<br>%{{y:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(**_base_layout("IoU distribution — comparison"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="IoU", range=[0, 1], dtick=0.1)
    fig.update_yaxes(title="% of samples", range=[0, y_max] if y_max else None)
    fig.update_layout(legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98))
    return fig


def generate_plots(results_dir: Path, y_max: float | None = None) -> list[Path]:
    """Generate all single-run plots for ``results_dir``. Returns image paths."""
    results = load_results(results_dir)
    if not results:
        raise ValueError(f"No results found in {results_dir}")

    label = _run_label(results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    figures = {
        "iou_histogram": plot_iou_histogram(results, label, y_max=y_max),
    }

    written: list[Path] = []
    for name, fig in figures.items():
        written.append(_save(fig, out_dir, name))
        # Interactive HTML alongside the PNG for exploration.
        fig.write_html(str(out_dir / f"{name}.html"), include_plotlyjs="cdn")

    return written


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m assistant.experiments.plots <results_dir> [...]")
        sys.exit(1)

    dirs = [Path(a) for a in sys.argv[1:]]

    # First pass: load every run so we can fix a single shared y-axis scale,
    # making all histograms directly comparable.
    runs: list[tuple[Path, str, list[dict]]] = []
    for results_dir in dirs:
        if not results_dir.exists():
            print(f"Directory not found: {results_dir}")
            continue
        results = load_results(results_dir)
        if not results:
            print(f"No results found in {results_dir}")
            continue
        runs.append((results_dir, _run_label(results_dir), results))

    if not runs:
        return

    shared_y_max = _Y_HEADROOM * max(_hist_peak_percent(r) for _, _, r in runs)

    for results_dir, _, _ in runs:
        written = generate_plots(results_dir, y_max=shared_y_max)
        print(f"{results_dir.name}: wrote {len(written)} plots to {written[0].parent}")

    if len(runs) > 1:
        out_dir = runs[0][0] / "plots"
        out_dir.mkdir(exist_ok=True)
        fig = plot_run_comparison(
            [(label, results) for _, label, results in runs], y_max=shared_y_max
        )
        _save(fig, out_dir, "comparison_iou_histogram")
        fig.write_html(
            str(out_dir / "comparison_iou_histogram.html"), include_plotlyjs="cdn"
        )
        print(f"comparison: wrote cross-run chart to {out_dir}")


if __name__ == "__main__":
    main()
