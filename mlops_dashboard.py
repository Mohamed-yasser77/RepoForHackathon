"""
mlops_dashboard.py
==================
Generate rich visualisations from MLflow experiment data.

Usage:
    python mlops_dashboard.py                    # latest run
    python mlops_dashboard.py --run-id <ID>      # specific run
    python mlops_dashboard.py --compare          # compare all runs in experiment

Charts are saved to mlruns_charts/ and opened automatically.
"""

import argparse
import os
import sys
import webbrowser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from classifier import CATEGORIES, MLFLOW_EXPERIMENT, CHARTS_DIR

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_client():
    return MlflowClient()


def _get_experiment_id(client):
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        print(f"[Dashboard] Experiment '{MLFLOW_EXPERIMENT}' not found.")
        print("  → Run 'python download_and_train.py' or 'python run_experiments.py' first.")
        sys.exit(1)
    return exp.experiment_id


def _get_runs(client, experiment_id, max_results=100):
    return client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.weighted_f1 DESC"],
        max_results=max_results,
    )


def _get_run(client, run_id):
    return client.get_run(run_id)


# ─── Chart 1: Per-class metrics for a single run ─────────────────────────────

def chart_per_class_metrics(run, save_dir=CHARTS_DIR):
    """Grouped bar chart: Precision / Recall / F1 per category."""
    metrics = run.data.metrics
    cats = CATEGORIES

    prec = [metrics.get(f"{c}_precision", 0) for c in cats]
    rec  = [metrics.get(f"{c}_recall",    0) for c in cats]
    f1   = [metrics.get(f"{c}_f1",        0) for c in cats]

    x = np.arange(len(cats))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w, prec, w, label="Precision", color="#3B82F6", edgecolor="white")
    b2 = ax.bar(x,     rec,  w, label="Recall",    color="#10B981", edgecolor="white")
    b3 = ax.bar(x + w, f1,   w, label="F1-Score",  color="#F59E0B", edgecolor="white")

    # Value labels on bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Precision / Recall / F1", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = os.path.join(save_dir, "dashboard_per_class.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ─── Chart 2: Overall metrics radar / summary ────────────────────────────────

def chart_overall_summary(run, save_dir=CHARTS_DIR):
    """Horizontal bar chart of overall metrics."""
    m = run.data.metrics
    labels = ["Accuracy", "Weighted F1", "Weighted Precision", "Weighted Recall",
              "Macro F1", "Macro Precision", "Macro Recall"]
    keys   = ["accuracy", "weighted_f1", "weighted_precision", "weighted_recall",
              "macro_f1", "macro_precision", "macro_recall"]
    vals = [m.get(k, 0) for k in keys]

    colors = sns.color_palette("viridis", len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, vals, color=colors, edgecolor="white")
    ax.set_xlim(0, 1.1)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=10)
    ax.set_title("Overall Model Metrics", fontsize=14, fontweight="bold")
    ax.set_xlabel("Score")
    fig.tight_layout()

    path = os.path.join(save_dir, "dashboard_overall.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ─── Chart 3: Experiment comparison ──────────────────────────────────────────

def chart_experiment_comparison(runs, save_dir=CHARTS_DIR):
    """Compare weighted_f1 and accuracy across multiple runs."""
    if not runs:
        print("  ⚠ No runs to compare.")
        return None

    run_names = []
    wf1_vals  = []
    acc_vals  = []

    for r in runs[:15]:  # cap at 15 for readability
        name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
        run_names.append(name)
        wf1_vals.append(r.data.metrics.get("weighted_f1", 0))
        acc_vals.append(r.data.metrics.get("accuracy", 0))

    x = np.arange(len(run_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(run_names) * 1.2), 6))
    ax.bar(x - w/2, wf1_vals, w, label="Weighted F1", color="#6366F1", edgecolor="white")
    ax.bar(x + w/2, acc_vals, w, label="Accuracy",    color="#EC4899", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Experiment Comparison: Weighted F1 vs Accuracy", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(save_dir, "dashboard_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ─── Chart 4: Hyperparameter impact analysis ─────────────────────────────────

def chart_hyperparam_impact(runs, save_dir=CHARTS_DIR):
    """Scatter plots showing how key hyperparams affect weighted F1."""
    if len(runs) < 3:
        print("  ⚠ Need at least 3 runs for hyperparameter impact analysis.")
        return None

    params_to_plot = ["w_tfidf_lr", "logreg_C", "bm25_k1", "tfidf_max_features"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, param in zip(axes, params_to_plot):
        xs = []
        ys = []
        for r in runs:
            val = r.data.params.get(param)
            wf1 = r.data.metrics.get("weighted_f1")
            if val is not None and wf1 is not None:
                try:
                    xs.append(float(val))
                    ys.append(float(wf1))
                except ValueError:
                    pass
        if xs:
            ax.scatter(xs, ys, alpha=0.7, s=60, c="#6366F1", edgecolors="white")
            ax.set_xlabel(param, fontsize=11)
            ax.set_ylabel("Weighted F1", fontsize=11)
            ax.set_title(f"Impact of {param}", fontsize=12, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Impact of {param}")

    fig.suptitle("Hyperparameter Impact on Weighted F1", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = os.path.join(save_dir, "dashboard_hyperparam_impact.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ─── Chart 5: Category distribution (from params) ────────────────────────────

def chart_category_distribution(run, save_dir=CHARTS_DIR):
    """Pie chart showing training data distribution by category."""
    params = run.data.params
    sizes = []
    labels_list = []
    for cat in CATEGORIES:
        val = params.get(f"corpus_{cat}", "0")
        sizes.append(int(val))
        labels_list.append(cat)

    if sum(sizes) == 0:
        print("  ⚠ No corpus distribution data found in run params.")
        return None

    colors = ["#3B82F6", "#10B981", "#F59E0B"]
    explode = [0.03] * len(CATEGORIES)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels_list, autopct="%1.1f%%",
        colors=colors, explode=explode, startangle=140,
        textprops={"fontsize": 12},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title("Training Data Distribution by Category", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(save_dir, "dashboard_category_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLOps Dashboard — visualise experiment results")
    parser.add_argument("--run-id", type=str, help="Specific MLflow run ID to visualise")
    parser.add_argument("--compare", action="store_true", help="Compare all runs in the experiment")
    parser.add_argument("--open", action="store_true", default=True,
                        help="Open generated charts (default: True)")
    args = parser.parse_args()

    os.makedirs(CHARTS_DIR, exist_ok=True)
    client = _get_client()
    exp_id = _get_experiment_id(client)

    print("="*65)
    print("  Smart-Support MVR — MLOps Dashboard")
    print("="*65)

    saved_paths = []

    if args.run_id:
        # Single run
        run = _get_run(client, args.run_id)
        run_name = run.data.tags.get("mlflow.runName", args.run_id[:8])
        print(f"\n  Visualising run: {run_name} ({args.run_id})\n")

        saved_paths.append(chart_per_class_metrics(run))
        saved_paths.append(chart_overall_summary(run))
        saved_paths.append(chart_category_distribution(run))
    else:
        # Latest run or comparison
        runs = _get_runs(client, exp_id)
        if not runs:
            print("  No runs found. Train a model first.")
            return

        latest = runs[0]
        run_name = latest.data.tags.get("mlflow.runName", latest.info.run_id[:8])
        print(f"\n  Latest run: {run_name} ({latest.info.run_id})")
        print(f"  Total runs in experiment: {len(runs)}\n")

        saved_paths.append(chart_per_class_metrics(latest))
        saved_paths.append(chart_overall_summary(latest))
        saved_paths.append(chart_category_distribution(latest))

        if args.compare or len(runs) > 1:
            saved_paths.append(chart_experiment_comparison(runs))
            saved_paths.append(chart_hyperparam_impact(runs))

    # Summary
    saved_paths = [p for p in saved_paths if p is not None]
    print(f"\n  📊 Generated {len(saved_paths)} charts in: {CHARTS_DIR}")

    if args.open and saved_paths:
        for p in saved_paths:
            try:
                webbrowser.open(f"file:///{os.path.abspath(p)}")
            except Exception:
                pass

    print(f"\n  To view in MLflow UI:")
    print(f"    mlflow ui --port 5000")
    print(f"    → http://localhost:5000\n")


if __name__ == "__main__":
    main()
