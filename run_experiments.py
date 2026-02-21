"""
run_experiments.py
==================
Automated hyperparameter sweep with MLflow experiment tracking.
Each configuration trains an 80/20 split, logs all metrics/params/artifacts.

Usage:
    python run_experiments.py           # full sweep (all combos)
    python run_experiments.py --quick   # quick sweep (4 configs, for testing)
"""

import itertools
import sys
import os
import time
from dataclasses import asdict

from dotenv import load_dotenv
import mlflow

from classifier import (
    EnsembleIRClassifier,
    HyperParams,
    load_dataset,
    CATEGORIES,
    MLFLOW_EXPERIMENT,
    CHARTS_DIR,
    MODEL_PATH,
)


# ─── Sweep Configurations ────────────────────────────────────────────────────

FULL_GRID = {
    "w_tfidf_lr":        [0.50, 0.60, 0.65, 0.70],
    "logreg_C":          [1.0, 5.0, 10.0],
    "tfidf_max_features": [30_000, 50_000],
    "bm25_k1":           [1.2, 1.5, 2.0],
}

QUICK_GRID = {
    "w_tfidf_lr":        [0.60, 0.70],
    "logreg_C":          [5.0, 10.0],
    "tfidf_max_features": [50_000],
    "bm25_k1":           [1.5],
}


def _build_configs(grid: dict) -> list[HyperParams]:
    """Expand a grid dict into a list of HyperParams objects."""
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    configs = []
    for combo in combos:
        overrides = dict(zip(keys, combo))
        # Ensure weights sum to 1: bm25 = 1 - tfidf_lr (bim stays at 0)
        w_tfidf = overrides.get("w_tfidf_lr", 0.65)
        overrides["w_bm25"] = round(1.0 - w_tfidf, 2)
        overrides["w_bim"]  = 0.0
        configs.append(HyperParams(**overrides))
    return configs


def run_sweep(configs: list[HyperParams], texts, labels):
    """Run all configs, log to MLflow, return best run info."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = []
    total = len(configs)

    for i, hp in enumerate(configs, 1):
        run_name = (
            f"sweep-w{hp.w_tfidf_lr:.2f}-C{hp.logreg_C}-"
            f"feat{hp.tfidf_max_features}-k1{hp.bm25_k1}"
        )
        print(f"\n{'='*65}")
        print(f"  Experiment {i}/{total}: {run_name}")
        print(f"{'='*65}")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(asdict(hp))
            mlflow.log_param("corpus_size", len(texts))
            mlflow.log_param("sweep_config", run_name)

            clf = EnsembleIRClassifier(hp=hp)
            metrics = clf.evaluate(texts, labels, log_mlflow=True)

            run_id = mlflow.active_run().info.run_id
            results.append({
                "run_id":   run_id,
                "run_name": run_name,
                "hp":       hp,
                "metrics":  metrics,
            })

            print(f"  → weighted_f1={metrics['weighted_f1']:.4f}  "
                  f"accuracy={metrics['accuracy']:.4f}")

    return results


def select_best(results) -> dict:
    """Pick the run with the highest weighted F1."""
    return max(results, key=lambda r: r["metrics"]["weighted_f1"])


def retrain_champion(hp: HyperParams, texts, labels):
    """Retrain on the full dataset with the champion config."""
    print("\n" + "="*65)
    print("  Retraining CHAMPION on full dataset")
    print("="*65)

    with mlflow.start_run(run_name="champion-full-retrain"):
        mlflow.log_params(asdict(hp))
        mlflow.log_param("corpus_size", len(texts))
        mlflow.log_param("is_champion", True)
        mlflow.set_tag("champion", "true")

        clf = EnsembleIRClassifier(hp=hp)
        metrics = clf.evaluate(texts, labels, log_mlflow=True)
        clf.train(texts, labels)
        clf.save()
        mlflow.log_artifact(MODEL_PATH)

        print(f"\n  ✓ Champion model saved to {MODEL_PATH}")
        print(f"  ✓ weighted_f1={metrics['weighted_f1']:.4f}")


def main():
    load_dotenv()
    quick = "--quick" in sys.argv

    grid = QUICK_GRID if quick else FULL_GRID
    configs = _build_configs(grid)

    print("="*65)
    print(f"  Smart-Support MVR — Experiment Sweep ({'QUICK' if quick else 'FULL'})")
    print(f"  Configurations: {len(configs)}")
    print("="*65)

    texts, labels = load_dataset()

    t0 = time.time()
    results = run_sweep(configs, texts, labels)
    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  SWEEP RESULTS SUMMARY")
    print("="*65)
    print(f"  Total runs:    {len(results)}")
    print(f"  Total time:    {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Sort by weighted F1 descending
    results.sort(key=lambda r: r["metrics"]["weighted_f1"], reverse=True)
    print(f"\n  {'Rank':<5} {'weighted_f1':<13} {'accuracy':<10} {'Config'}")
    print(f"  {'─'*5} {'─'*13} {'─'*10} {'─'*40}")
    for rank, r in enumerate(results[:10], 1):
        m = r["metrics"]
        print(f"  {rank:<5} {m['weighted_f1']:<13.4f} {m['accuracy']:<10.4f} {r['run_name']}")

    best = select_best(results)
    print(f"\n  🏆 BEST RUN: {best['run_name']}")
    print(f"     weighted_f1 = {best['metrics']['weighted_f1']:.4f}")
    print(f"     accuracy    = {best['metrics']['accuracy']:.4f}")
    print(f"     run_id      = {best['run_id']}")

    # Retrain champion on full data
    retrain_champion(best["hp"], texts, labels)

    print(f"\n  ✓ All done! View experiments with:")
    print(f"    cd {os.path.dirname(os.path.abspath(__file__))}")
    print(f"    mlflow ui --port 5000")
    print(f"    → http://localhost:5000\n")


if __name__ == "__main__":
    main()
