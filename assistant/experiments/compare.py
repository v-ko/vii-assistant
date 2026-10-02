"""Hand-curated comparison charts for the thesis.

Unlike ``plots.py`` (which auto-derives charts from a results dir), this script
holds the numbers we actually want to present *by hand*, so the labels, grouping
and ordering are exactly what goes into the document. Edit ``RUNS`` below.

Outputs vertical grouped bar charts (PNG + HTML) into ``experiments/comparisons/``
at the project root.

Run:
    /sync/projects/misli/dev_venv/bin/python -m assistant.experiments.compare
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

# --- Hardcoded results -------------------------------------------------------
# label -> metrics. Percentages are 0..100, IoU is 0..1.
# Edit these by hand as new runs come in.

RUNS: dict[str, dict[str, float]] = {
    # Localization only, JSON format, no extra context (i2e_bench_localization 021435).
    "Локализация (без допълнителен контекст)": {
        "mean_iou": 0.4077,
        "median_iou": 0.4419,
        "acc@0.3": 54.84,
        "acc@0.5": 46.45,
        "acc@0.7": 31.01,
    },
    # Python code generation, no focus mode (i2e_bench_tool 041152).
    "Локализация с python код (без фокусен режим)": {
        "mean_iou": 0.3664,
        "median_iou": 0.3074,
        "acc@0.3": 50.30,
        "acc@0.5": 40.76,
        "acc@0.7": 26.61,
    },
    # Python code generation with localization focus mode (i2e_bench_tool 082050).
    "Локализация с python код и фокусен режим": {
        "mean_iou": 0.4290,
        "median_iou": 0.4714,
        "acc@0.3": 57.48,
        "acc@0.5": 48.41,
        "acc@0.7": 32.63,
    },
}

# Synthetic-rectangle base-capability comparison (section 4.1).
# Both models evaluated over the full 200-sample synthetic set.
SYNTHETIC_RUNS: dict[str, dict[str, float]] = {
    "Qwen3-VL-4B": {
        "mean_iou": 0.9292,
        "median_iou": 0.9403,
        "acc@0.3": 100.00,
        "acc@0.5": 100.00,
        "acc@0.7": 100.00,
    },
    "Gemma 4 E4B (GGUF Q8_0)": {
        "mean_iou": 0.1287,
        "median_iou": 0.0223,
        "acc@0.3": 18.00,
        "acc@0.5": 3.50,
        "acc@0.7": 0.00,
    },
}

_PALETTE = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#dc2626", "#0891b2"]

_IMG_WIDTH = 800
_IMG_HEIGHT = 500
_IMG_SCALE = 2

_OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "comparisons"


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)),
        template="plotly_white",
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=14),
        margin=dict(l=70, r=40, t=70, b=60),
        width=_IMG_WIDTH,
        height=_IMG_HEIGHT,
        barmode="group",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
    )


def _grouped_bars(
    runs: dict[str, dict[str, float]],
    metrics: list[tuple[str, str]],  # (key, x-axis label)
    title: str,
    *,
    percent: bool,
) -> go.Figure:
    """One bar group per metric, one colored bar per run."""
    fig = go.Figure()
    labels = [lbl for _, lbl in metrics]
    keys = [k for k, _ in metrics]
    for idx, run in enumerate(runs):
        values = [runs[run][k] for k in keys]
        text = [f"{v:.1f}%" if percent else f"{v:.3f}" for v in values]
        fig.add_trace(
            go.Bar(
                name=run,
                x=labels,
                y=values,
                marker_color=_PALETTE[idx % len(_PALETTE)],
                text=text,
                textposition="outside",
                cliponaxis=False,
            )
        )
    fig.update_layout(**_base_layout(title))
    if percent:
        fig.update_yaxes(title="Точност (%)", range=[0, 100])
    else:
        fig.update_yaxes(title="IoU", range=[0, 1])
    return fig


def _save(fig: go.Figure, name: str) -> Path:
    _OUT_DIR.mkdir(exist_ok=True)
    png = _OUT_DIR / f"{name}.png"
    fig.write_image(str(png), scale=_IMG_SCALE)
    fig.write_html(str(_OUT_DIR / f"{name}.html"), include_plotlyjs="cdn")
    return png


def main() -> None:
    written = []
    written.append(
        _save(
            _grouped_bars(
                RUNS,
                [
                    ("acc@0.3", "Acc@0.3"),
                    ("acc@0.5", "Acc@0.5"),
                    ("acc@0.7", "Acc@0.7"),
                ],
                "Точност по праг на IoU",
                percent=True,
            ),
            "accuracy_comparison",
        )
    )
    written.append(
        _save(
            _grouped_bars(
                RUNS,
                [("mean_iou", "Средно IoU"), ("median_iou", "Медианно IoU")],
                "Сравнение по IoU",
                percent=False,
            ),
            "iou_comparison",
        )
    )
    written.append(
        _save(
            _grouped_bars(
                SYNTHETIC_RUNS,
                [
                    ("acc@0.3", "Acc@0.3"),
                    ("acc@0.5", "Acc@0.5"),
                    ("acc@0.7", "Acc@0.7"),
                ],
                "Синтетична локализация — точност по праг на IoU",
                percent=True,
            ),
            "synthetic_accuracy_comparison",
        )
    )
    written.append(
        _save(
            _grouped_bars(
                SYNTHETIC_RUNS,
                [("mean_iou", "Средно IoU"), ("median_iou", "Медианно IoU")],
                "Синтетична локализация — сравнение по IoU",
                percent=False,
            ),
            "synthetic_iou_comparison",
        )
    )
    print(f"Wrote {len(written)} charts to {_OUT_DIR}")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
