"""Matplotlib helpers: Kaplan-Meier curves, calibration, performance comparison.

Style: thin lines, recessive grid, fixed colour order (never cycled), legend whenever
there are two or more series. Colours are the first slots of a colour-vision-safe
categorical palette (light-surface values).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# categorical slots in fixed order: blue, orange, aqua, yellow, magenta, green, violet, red
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "font.family": "sans-serif", "font.size": 9.5,
        "lines.linewidth": 1.6, "legend.frameon": False,
        "figure.dpi": 110,
    })


def save(fig, name: str) -> Path:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


# --- Kaplan-Meier ------------------------------------------------------------
def plot_km(df: pd.DataFrame, group: str, ax=None, order: list[str] | None = None,
            title: str | None = None, years: bool = True):
    """Kaplan-Meier curves by `group` (df needs time [months], event, and the group column)."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    apply_style()
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    levels = order or sorted(df[group].dropna().unique())
    scale = 12.0 if years else 1.0
    for i, lvl in enumerate(levels):
        sub = df[df[group] == lvl]
        if len(sub) < 5:
            continue
        kmf = KaplanMeierFitter().fit(sub["time"] / scale, sub["event"], label=f"{lvl} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=CATEGORICAL[i % len(CATEGORICAL)],
                                   show_censors=True, censor_styles={"ms": 3, "marker": "|"})
    sub = df[df[group].isin(levels)].dropna(subset=[group])
    p = multivariate_logrank_test(sub["time"], sub[group], sub["event"]).p_value
    ax.set_xlabel("Годы" if years else "Месяцы")
    ax.set_ylabel("Вероятность выживания")
    ax.set_ylim(0, 1.02)
    ax.set_title(title or f"Каплан–Мейер: {group}  (log-rank p = {p:.3g})")
    ax.legend(loc="lower left", fontsize=8.5)
    return ax


# --- calibration -------------------------------------------------------------
def plot_calibration(tables: dict[str, pd.DataFrame], t_months: float, ax=None):
    """`tables`: label -> output of evaluate.calibration_table. Points on the diagonal = well calibrated."""
    apply_style()
    ax = ax or plt.subplots(figsize=(4.6, 4.6))[1]
    hi = max(0.05, max(max(t["predicted"].max(), t["observed"].max()) for t in tables.values()) * 1.15)
    ax.plot([0, hi], [0, hi], color=MUTED, lw=0.9, ls="--", label="идеальная калибровка")
    for i, (label, t) in enumerate(tables.items()):
        ax.plot(t["predicted"], t["observed"], marker="o", ms=5, color=CATEGORICAL[i],
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(f"Предсказанный риск события за {t_months / 12:.0f} лет")
    ax.set_ylabel("Наблюдаемый риск (Каплан–Мейер)")
    ax.set_title("Калибровка по группам риска")
    ax.legend(fontsize=8.5, loc="upper left")
    return ax


# --- performance comparison --------------------------------------------------
def plot_performance(internal: pd.DataFrame, external: pd.DataFrame):
    """Two panels, one row per model: Harrell's C (left) and Uno's C (right).

    internal: columns features, model, harrell_c, uno_c (mean over CV folds)
    external: columns features, model, harrell_c, ci_low, ci_high, uno_c
    Harrell's C depends on follow-up length and censoring, so the two cohorts are only
    comparable on Uno's C; both are shown on purpose.
    """
    apply_style()
    key = lambda d: d["features"] + " / " + d["model"]
    a = internal.assign(label=key(internal)).set_index("label")
    b = external.assign(label=key(external)).set_index("label")
    labels = [l for l in a.index if l in b.index]
    y = np.arange(len(labels))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.55 * len(labels) + 1.8), sharey=True)
    for ax, metric, title in ((axes[0], "harrell_c", "C-индекс Харрела"),
                              (axes[1], "uno_c", "C-индекс Уно (устойчив к цензурированию)")):
        ax.axvline(0.5, color=MUTED, lw=0.9, ls="--")
        ax.scatter(a.loc[labels, metric], y, s=42, color=CATEGORICAL[0], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label="TCGA, внутренняя CV (среднее)")
        if metric == "harrell_c":
            ax.errorbar(b.loc[labels, metric], y,
                        xerr=[b.loc[labels, metric] - b.loc[labels, "ci_low"],
                              b.loc[labels, "ci_high"] - b.loc[labels, metric]],
                        fmt="o", ms=6, color=CATEGORICAL[1], ecolor=CATEGORICAL[1], elinewidth=1.2,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                        label="METABRIC, внешняя (95% бутстреп-ИД)")
        else:
            ax.scatter(b.loc[labels, metric], y, s=48, color=CATEGORICAL[1], edgecolor=SURFACE,
                       linewidth=1.5, zorder=3, label="METABRIC, внешняя")
        ax.set_yticks(y, labels)
        ax.set_xlabel("0,5 = случайное угадывание")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Переносится ли качество модели на новую когорту?", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return fig
