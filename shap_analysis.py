# ═══════════════════════════════════════════════════════════════
#  shap_analysis.py
#  Explainable AI using SHAP for Network Traffic Anomaly Detection
#  Analyzes: Random Forest (best model) + Logistic Regression
#  Datasets: UNSW-NB15, CICIDS2017, CICIDS2018
#  Author: Piyush Bhati | RVU Summer Internship 2026
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Output folder ─────────────────────────────────────────────
SHAP_DIR = "results/plots/shap"
os.makedirs(SHAP_DIR, exist_ok=True)

# ── Dataset configs ───────────────────────────────────────────
DATASETS = {
    "UNSW-NB15": {
        "train": "data/processed/train_cleaned.csv",
        "test":  "data/processed/test_cleaned.csv",
        "rf_model": "models/local/random_forest_baseline.pkl",
        "lr_model": "models/local/logistic_regression_baseline.pkl",
        "sample": 500,   # SHAP is slow — use small sample
    },
    "CICIDS2017": {
        "train": "data/processed/cicids2017_train.csv",
        "test":  "data/processed/cicids2017_test.csv",
        "rf_model": "models/local/cicids2017_rf.pkl",
        "lr_model": "models/local/cicids2017_lr.pkl",
        "sample": 500,
    },
    "CICIDS2018": {
        "train": "data/processed/cicids2018_train.csv",
        "test":  "data/processed/cicids2018_test.csv",
        "rf_model": "models/local/cicids2018_rf.pkl",
        "lr_model": "models/local/cicids2018_lr.pkl",
        "sample": 500,
    },
}


def load_data(cfg, sample=500):
    """Load test data and take a small sample for SHAP."""
    test_df  = pd.read_csv(cfg["test"])
    X_test   = test_df.drop(columns=["label"])
    y_test   = test_df["label"]

    # Sample equally from normal and attack
    n_each  = sample // 2
    normal  = test_df[test_df["label"] == 0].sample(
        min(n_each, (test_df["label"]==0).sum()), random_state=42)
    attack  = test_df[test_df["label"] == 1].sample(
        min(n_each, (test_df["label"]==1).sum()), random_state=42)
    sample_df = pd.concat([normal, attack]).reset_index(drop=True)

    X_sample = sample_df.drop(columns=["label"])
    y_sample = sample_df["label"]
    return X_test, y_test, X_sample, y_sample


def run_shap_rf(dataset_name, cfg):
    """SHAP analysis for Random Forest model."""
    print(f"\n  🌲 Random Forest SHAP — {dataset_name}")

    # Load model and data
    rf = joblib.load(cfg["rf_model"])
    X_test, y_test, X_sample, y_sample = load_data(
        cfg, cfg["sample"])

    feature_names = list(X_sample.columns)

    print(f"    Computing SHAP values for {len(X_sample)} samples...")
    # TreeExplainer is fast for Random Forest
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values is a list [class0, class1]
    # We want class 1 (attack) values
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Newer SHAP versions return (n_samples, n_features, n_classes)
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    sv = np.array(sv)
    if sv.ndim != 2:
        sv = sv.reshape(X_sample.shape[0], -1)

    prefix = dataset_name.lower().replace("-","").replace(" ","_")

    # ── Plot 1: Feature Importance Bar Chart ──────────────────
    print(f"    📊 Saving feature importance bar chart...")
    mean_abs = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs
    }).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df["feature"][::-1],
                   importance_df["importance"][::-1],
                   color="#2E75B6", edgecolor="black")
    ax.set_xlabel("Mean |SHAP Value| (Feature Importance)",
                  fontweight="bold")
    ax.set_title(f"Top 15 Most Important Features\n"
                 f"Random Forest — {dataset_name}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path1 = f"{SHAP_DIR}/{prefix}_rf_feature_importance.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Saved: {path1}")

    # ── Plot 2: SHAP Summary Plot (Beeswarm) ──────────────────
    print(f"    📊 Saving SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample,
                      feature_names=feature_names,
                      max_display=15,
                      show=False)
    plt.title(f"SHAP Summary Plot — Random Forest — {dataset_name}",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path2 = f"{SHAP_DIR}/{prefix}_rf_summary_plot.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Saved: {path2}")

    # ── Plot 3: SHAP Waterfall (one attack sample) ────────────
    print(f"    📊 Saving SHAP waterfall plot...")
    # Find first attack sample
    attack_idx = y_sample[y_sample == 1].index[0]
    pos        = X_sample.index.get_loc(attack_idx)

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.atleast_1d(ev)
        base_val = float(ev[1]) if len(ev) > 1 else float(ev[0])
    else:
        base_val = float(ev)

    shap_exp = shap.Explanation(
        values=sv[pos],
        base_values=base_val,
        data=X_sample.iloc[pos].values,
        feature_names=feature_names
    )
    plt.figure(figsize=(10, 7))
    shap.waterfall_plot(shap_exp, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Single Attack Sample\n"
              f"Random Forest — {dataset_name}",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    path3 = f"{SHAP_DIR}/{prefix}_rf_waterfall.png"
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Saved: {path3}")

    # Top 5 features
    top5 = importance_df.head(5)["feature"].tolist()
    print(f"\n    🏆 Top 5 most important features for {dataset_name}:")
    for i, (f, v) in enumerate(zip(
            importance_df.head(5)["feature"],
            importance_df.head(5)["importance"]), 1):
        print(f"      {i}. {f:<30} SHAP={v:.4f}")

    return importance_df


def run_shap_lr(dataset_name, cfg):
    """SHAP analysis for Logistic Regression model."""
    print(f"\n  📈 Logistic Regression SHAP — {dataset_name}")

    lr = joblib.load(cfg["lr_model"])
    X_test, y_test, X_sample, y_sample = load_data(
        cfg, cfg["sample"])

    feature_names = list(X_sample.columns)

    print(f"    Computing SHAP values for {len(X_sample)} samples...")
    # LinearExplainer is fast for Logistic Regression
    explainer   = shap.LinearExplainer(lr, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # For binary LR, shap_values shape is (n_samples, n_features)
    if len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    prefix = dataset_name.lower().replace("-","").replace(" ","_")

    # Feature importance
    mean_abs = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs
    }).sort_values("importance", ascending=False).head(15)

    # ── Plot: Feature Importance ──────────────────────────────
    print(f"    📊 Saving LR feature importance chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"][::-1],
            importance_df["importance"][::-1],
            color="#D85A30", edgecolor="black")
    ax.set_xlabel("Mean |SHAP Value|", fontweight="bold")
    ax.set_title(f"Top 15 Features — Logistic Regression — {dataset_name}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = f"{SHAP_DIR}/{prefix}_lr_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Saved: {path}")

    # Summary plot
    print(f"    📊 Saving LR summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample,
                      feature_names=feature_names,
                      max_display=15, show=False)
    plt.title(f"SHAP Summary — LR — {dataset_name}",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path2 = f"{SHAP_DIR}/{prefix}_lr_summary_plot.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Saved: {path2}")

    top5 = importance_df.head(5)["feature"].tolist()
    print(f"\n    🏆 Top 5 features (LR) for {dataset_name}:")
    for i,(f,v) in enumerate(zip(
            importance_df.head(5)["feature"],
            importance_df.head(5)["importance"]),1):
        print(f"      {i}. {f:<30} SHAP={v:.4f}")

    return importance_df


def compare_datasets(all_rf_importance):
    """Compare top features across all 3 datasets."""
    print(f"\n{'='*60}")
    print(f"  📊 CROSS-DATASET FEATURE IMPORTANCE COMPARISON")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    colors = ["#2E75B6", "#27500A", "#D85A30"]

    for ax, (ds_name, imp_df), color in zip(
            axes, all_rf_importance.items(), colors):
        top10 = imp_df.head(10)
        ax.barh(top10["feature"][::-1],
                top10["importance"][::-1],
                color=color, edgecolor="black")
        ax.set_title(f"{ds_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value|")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Top 10 Most Important Features — RF SHAP\n"
                 "Comparison Across All 3 Datasets",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{SHAP_DIR}/cross_dataset_feature_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SHAP EXPLAINABILITY ANALYSIS")
    print("  Federated Learning - Network Traffic Anomaly Detection")
    print("=" * 60)

    all_rf_importance = {}

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*60}")

        # Random Forest SHAP
        try:
            rf_imp = run_shap_rf(ds_name, ds_cfg)
            all_rf_importance[ds_name] = rf_imp
        except Exception as e:
            print(f"  ❌ RF SHAP failed: {e}")

        # Logistic Regression SHAP
        try:
            run_shap_lr(ds_name, ds_cfg)
        except Exception as e:
            print(f"  ❌ LR SHAP failed: {e}")

    # Cross-dataset comparison
    if len(all_rf_importance) == 3:
        compare_datasets(all_rf_importance)

    print(f"\n{'='*60}")
    print(f"  ✅ SHAP ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"  Plots saved in: results/plots/shap/")
    print(f"  Files generated:")
    for f in os.listdir(SHAP_DIR):
        print(f"    ✅ {f}")
    print(f"\n👉 NEXT: Add SHAP plots to your report!")