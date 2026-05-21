"""Matplotlib/seaborn figures for modeling notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .models.dataclasses import ProfitCurveResult, TargetingSelectionResult
from .models.profit_targeting import expected_value_per_contact


def plot_topk_oof_profit_curve(
    profit_curve: ProfitCurveResult,
    feature_count: int,
    max_targets: int,
) -> plt.Figure:
    """OOF business score vs k with optimal-k marker."""
    curve = profit_curve.curve
    color = sns.color_palette("husl")[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        x="k",
        y="score",
        data=curve,
        linewidth=2.5,
        color=color,
        ax=ax,
        label="Business score (OOF)",
    )
    ax.axvline(
        profit_curve.best_k,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"Optimal k={profit_curve.best_k} (score={profit_curve.best_score:.0f})",
    )
    ax.fill_between(curve["k"], curve["score"], alpha=0.2, color=color)
    ax.set_xlabel("Targets selected (k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Business score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"OOF profit curve ({feature_count} feature(s), k ≤ {max_targets})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ev_targeting_dashboard(
    profit_curve: ProfitCurveResult,
    targeting: TargetingSelectionResult,
    oof_probabilities: np.ndarray,
    ev_rank_limit: int = 200,
) -> plt.Figure:
    """OOF profit curve and top-ranked expected value per contact."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(x="k", y="score", data=profit_curve.curve, ax=axes[0], linewidth=2)
    axes[0].axvline(
        targeting.selected_k,
        color="red",
        linestyle="--",
        label=f"k={targeting.selected_k}",
    )
    axes[0].axvline(
        targeting.k_ev_threshold,
        color="orange",
        linestyle=":",
        label=f"k_ev={targeting.k_ev_threshold}",
    )
    axes[0].set_title("OOF business score vs k")
    axes[0].legend()

    limit = min(ev_rank_limit, len(oof_probabilities))
    top_probs = oof_probabilities[np.argsort(oof_probabilities)[::-1][:limit]]
    ev_df = pd.DataFrame({
        "rank": np.arange(1, limit + 1),
        "expected_value": expected_value_per_contact(top_probs),
    })
    sns.lineplot(x="rank", y="expected_value", data=ev_df, ax=axes[1])
    axes[1].axhline(0.0, color="red", linestyle="--", label="break-even")
    axes[1].set_title(f"Top-{limit} OOF expected value per contact")
    axes[1].legend()

    fig.tight_layout()
    return fig
