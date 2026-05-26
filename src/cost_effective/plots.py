"""Matplotlib/seaborn figures for modeling notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

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


def plot_cluster_k_selection_grid(
    details: pd.DataFrame,
    *,
    ncols: int = 3,
) -> plt.Figure:
    """Inertia (elbow) and silhouette vs k, one panel per feature set."""
    feature_sets = sorted(details["feature_set"].unique())
    n = len(feature_sets)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    inertia_color, silhouette_color = sns.color_palette("husl")[:2]

    for idx, fs in enumerate(feature_sets):
        ax = axes[idx // ncols, idx % ncols]
        sub = details.loc[details["feature_set"] == fs].sort_values("k")
        ax2 = ax.twinx()
        ax.plot(sub["k"], sub["inertia"], "o-", color=inertia_color)
        ax2.plot(sub["k"], sub["silhouette"], "s-", color=silhouette_color)
        ax.set_title(fs)
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        ax2.set_ylabel("Silhouette")
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=inertia_color,
            marker="o",
            markersize=6,
            linestyle="-",
            label="Inertia (elbow)",
        ),
        Line2D(
            [0],
            [0],
            color=silhouette_color,
            marker="s",
            markersize=6,
            linestyle="-",
            label="Silhouette",
        ),
    ]
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle("KMeans k selection: inertia + silhouette", y=0.98)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=10,
        columnspacing=2.0,
        handletextpad=0.8,
        bbox_to_anchor=(0.5, 0.94),
    )
    return fig


def plot_cluster_y_profile(
    profile: pd.DataFrame,
    *,
    title: str,
    global_rate: float | None = None,
) -> plt.Figure:
    """Counts, positive rate, and stacked y=0/1 per cluster for one feature set."""
    baseline = global_rate
    if baseline is None and "global_positive_rate" in profile.columns:
        baseline = float(profile["global_positive_rate"].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_df = profile.sort_values("cluster").copy()
    cluster_ids = plot_df["cluster"].to_numpy(dtype=int)
    c0, c1 = sns.color_palette("husl", n_colors=2)

    axes[0].bar(cluster_ids, plot_df["n"], width=0.6, color=c0)
    axes[0].set_xticks(cluster_ids)
    axes[0].set_title("Cluster size")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Count")

    axes[1].bar(cluster_ids, plot_df["positive_rate"], width=0.6, color=c1)
    axes[1].set_xticks(cluster_ids)
    if baseline is not None:
        axes[1].axhline(baseline, color="red", linestyle="--", linewidth=1.5, label="Global rate")
        axes[1].legend(fontsize=8)
    axes[1].set_title("P(y=1) by cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Positive rate")
    axes[1].set_ylim(0, min(1.0, max(0.35, profile["positive_rate"].max() * 1.15)))

    stack = plot_df.set_index("cluster")[["n_negative", "n_positive"]]
    stack.plot(kind="bar", stacked=True, ax=axes[2], color=["#95a5a6", "#2ecc71"])
    axes[2].set_xticklabels([str(int(i)) for i in stack.index], rotation=0)
    axes[2].set_title("y=0 vs y=1")
    axes[2].set_xlabel("Cluster")
    axes[2].set_ylabel("Count")
    axes[2].legend(["y=0", "y=1"], fontsize=8)

    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_cluster_y_profiles_grid(
    profiles: pd.DataFrame,
    *,
    ncols: int = 3,
) -> plt.Figure:
    """Positive rate by cluster (one panel per feature set with chosen k)."""
    feature_sets = sorted(profiles["feature_set"].unique())
    n = len(feature_sets)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    bar_color = sns.color_palette("husl")[0]

    for idx, fs in enumerate(feature_sets):
        ax = axes[idx // ncols, idx % ncols]
        sub = profiles.loc[profiles["feature_set"] == fs].sort_values("cluster")
        baseline = float(sub["global_positive_rate"].iloc[0])
        k = int(sub["n_clusters"].iloc[0])
        cluster_ids = sub["cluster"].to_numpy(dtype=int)
        ax.bar(
            cluster_ids,
            sub["positive_rate"],
            width=0.6,
            color=bar_color,
        )
        ax.set_xticks(cluster_ids)
        ax.axhline(baseline, color="red", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_ylim(0, min(1.0, max(0.35, sub["positive_rate"].max() * 1.15)))
        ax.set_title(f"{fs} (k={k})")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("P(y=1)")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Target rate by cluster (red = global train rate)", y=1.02)
    fig.tight_layout()
    return fig


def plot_approaches_comparison(
    comparison: pd.DataFrame,
    *,
    metric: str = "oof_business_score",
    title: str | None = None,
) -> plt.Figure:
    """Bar chart of OOF (or CV) business score across modeling approaches."""
    if comparison.empty or metric not in comparison.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("No comparison data")
        return fig

    plot_df = comparison.dropna(subset=[metric]).copy()
    plot_df = plot_df.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("husl", n_colors=len(plot_df))
    sns.barplot(
        data=plot_df,
        x="label",
        y=metric,
        hue="label",
        palette=palette,
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Business score (penalized)")
    ax.set_title(title or f"Modeling approaches — {metric}")
    ax.tick_params(axis="x", rotation=25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_approaches_features_vs_oof(comparison: pd.DataFrame) -> plt.Figure:
    """Scatter: feature count vs OOF business score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if comparison.empty:
        ax.set_title("No comparison data")
        return fig
    plot_df = comparison.dropna(subset=["oof_business_score"])
    sns.scatterplot(
        data=plot_df,
        x="n_features",
        y="oof_business_score",
        hue="label",
        s=120,
        ax=ax,
    )
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["label"],
            (row["n_features"], row["oof_business_score"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_xlabel("Variables declared (submission)")
    ax.set_ylabel("OOF business score (penalized)")
    ax.set_title("Feature cost vs OOF performance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
