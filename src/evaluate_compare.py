"""
evaluate_compare.py
----------------------------------
Generates final comparison visuals + master metrics CSV for the report.

Produces:
  1. centralized_comparison.png   — bar chart: LR/RF/DNN across all 3 datasets
  2. federated_vs_centralized.png — centralized vs federated on UNSW-NB15
  3. fl_accuracy_progress.png     — FedAvg accuracy across 10 rounds (LR + DNN)
  4. master_metrics.csv           — all numbers in one table

Run from FL_Anomaly_Detection folder:
    python src/evaluate_compare.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR     = "results/plots"
METRICS_OUTPUT = "results/metrics/master_metrics.csv"

# ── 1. All centralized results ────────────────────────────────
centralized_results = [
    {"dataset": "UNSW-NB15",  "model": "Logistic Regression", "accuracy": 93.23, "precision": 91.73, "recall": 98.99, "f1": 0.9522},
    {"dataset": "UNSW-NB15",  "model": "Random Forest",       "accuracy": 96.07, "precision": 96.44, "recall": 97.84, "f1": 0.9714},
    {"dataset": "UNSW-NB15",  "model": "DNN",                 "accuracy": 93.35, "precision": 97.72, "recall": 92.39, "f1": 0.9498},
    {"dataset": "CICIDS2017", "model": "Logistic Regression", "accuracy": 96.69, "precision": 86.51, "recall": 92.43, "f1": 0.8937},
    {"dataset": "CICIDS2017", "model": "Random Forest",       "accuracy": 99.90, "precision": 99.70, "recall": 99.67, "f1": 0.9968},
    {"dataset": "CICIDS2017", "model": "DNN",                 "accuracy": 98.40, "precision": 94.28, "recall": 95.19, "f1": 0.9473},
    {"dataset": "CICIDS2018", "model": "Logistic Regression", "accuracy": 99.89, "precision": 99.25, "recall": 99.99, "f1": 0.9962},
    {"dataset": "CICIDS2018", "model": "Random Forest",       "accuracy": 100.00,"precision": 99.99, "recall": 100.00,"f1": 1.0000},
    {"dataset": "CICIDS2018", "model": "DNN",                 "accuracy": 99.98, "precision": 99.96, "recall": 99.92, "f1": 0.9994},
]

# ── 2. Federated vs Centralized (UNSW-NB15) ──────────────────
federated_results = [
    {"model": "Logistic Regression", "centralized": 93.23, "federated": 84.33},
    {"model": "Random Forest",       "centralized": 96.07, "federated": 95.39},
    {"model": "DNN",                 "centralized": 93.35, "federated": 93.91},
]

# ── 3. FedAvg round-by-round accuracy from your training logs ─
fl_rounds = list(range(0, 11))
fl_lr_accuracy  = [31.70, 68.06, 68.06, 74.75, 84.23,
                   84.33, 83.71, 82.89, 82.20, 81.74, 81.48]
fl_dnn_accuracy = [68.06, 93.14, 93.15, 93.16, 93.72,
                   93.68, 93.79, 93.64, 93.19, 93.91, 93.76]


# ── Plot 1: Centralized comparison across 3 datasets ─────────
def plot_centralized_comparison(df):
    print("  Plotting centralized comparison...")
    datasets = ["UNSW-NB15", "CICIDS2017", "CICIDS2018"]
    models   = ["Logistic Regression", "Random Forest", "DNN"]
    colors   = ["#2E75B6", "#27500A", "#D85A30"]
    x        = np.arange(len(datasets))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        values = []
        for d in datasets:
            row = df[(df.dataset == d) & (df.model == model)]
            values.append(row["accuracy"].values[0])
        bars = ax.bar(x + i * width, values, width,
                      label=model, color=color, edgecolor="black")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center",
                    fontsize=8, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Centralized Model Accuracy Across Datasets",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylim(80, 104)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "centralized_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {path}")


# ── Plot 2: Federated vs Centralized ─────────────────────────
def plot_federated_vs_centralized(df):
    print("  Plotting federated vs centralized...")
    models = df["model"].tolist()
    x      = np.arange(len(models))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, df["centralized"], width,
                   label="Centralized", color="#2E75B6",
                   edgecolor="black")
    bars2 = ax.bar(x + width/2, df["federated"],   width,
                   label="Federated (FedAvg / Tree Aggregation)",
                   color="#D85A30", edgecolor="black")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.2f}%",
                ha="center", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.2f}%",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Federated vs Centralized Accuracy (UNSW-NB15)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(75, 104)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "federated_vs_centralized.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {path}")


# ── Plot 3: FL accuracy progress over rounds ──────────────────
def plot_fl_progress():
    print("  Plotting FL training progress...")
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(fl_rounds, fl_lr_accuracy,  marker="o",
            color="#2E75B6", linewidth=2,
            markersize=6, label="Federated LR (FedAvg)")
    ax.plot(fl_rounds, fl_dnn_accuracy, marker="s",
            color="#D85A30", linewidth=2,
            markersize=6, label="Federated DNN (FedAvg)")

    ax.axhline(y=93.23, color="#2E75B6", linestyle="--",
               alpha=0.5, label="Centralized LR (93.23%)")
    ax.axhline(y=96.07, color="green",   linestyle="--",
               alpha=0.5, label="Centralized RF (96.07%)")
    ax.axhline(y=93.35, color="#D85A30", linestyle="--",
               alpha=0.5, label="Centralized DNN (93.35%)")

    ax.set_xlabel("Federated Learning Round", fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)",        fontweight="bold")
    ax.set_title("FedAvg Training Progress — Accuracy per Round",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(20, 100)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fl_accuracy_progress.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {path}")


# ── Save master metrics CSV ───────────────────────────────────
def save_master_metrics(centralized_df):
    print("  Saving master metrics CSV...")

    centralized_df["type"] = "Centralized"

    fed_rows = [
        {"dataset": "UNSW-NB15", "model": "Logistic Regression",
         "accuracy": 84.33, "precision": None, "recall": None,
         "f1": None, "type": "Federated (FedAvg)"},
        {"dataset": "UNSW-NB15", "model": "Random Forest",
         "accuracy": 95.39, "precision": 97.16, "recall": 96.04,
         "f1": 0.9660, "type": "Federated (Tree Aggregation)"},
        {"dataset": "UNSW-NB15", "model": "DNN",
         "accuracy": 93.91, "precision": None, "recall": None,
         "f1": None, "type": "Federated (FedAvg)"},
    ]

    master_df = pd.concat(
        [centralized_df, pd.DataFrame(fed_rows)],
        ignore_index=True
    )
    os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
    master_df.to_csv(METRICS_OUTPUT, index=False)
    print(f"  ✅ Saved: {METRICS_OUTPUT}")
    return master_df


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GENERATING COMPARISON CHARTS + MASTER METRICS")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    centralized_df = pd.DataFrame(centralized_results)
    federated_df   = pd.DataFrame(federated_results)

    print("\n📊 Chart 1: Centralized comparison (3 datasets)...")
    plot_centralized_comparison(centralized_df)

    print("\n📊 Chart 2: Federated vs Centralized (UNSW-NB15)...")
    plot_federated_vs_centralized(federated_df)

    print("\n📊 Chart 3: FL training progress over rounds...")
    plot_fl_progress()

    print("\n💾 Master metrics CSV...")
    master_df = save_master_metrics(centralized_df)

    print("\n" + "=" * 60)
    print("  ✅ ALL DONE!")
    print("=" * 60)
    print("\n📋 COMPLETE RESULTS SUMMARY:")
    print(f"\n  {'Type':<30} {'Model':<25} {'Accuracy':>10}")
    print(f"  {'-'*65}")
    for _, row in master_df.iterrows():
        print(f"  {row['type']:<30} {row['model']:<25} "
              f"{row['accuracy']:>9.2f}%")
    print("\n  Charts saved in: results/plots/")
    print("  Metrics saved in: results/metrics/master_metrics.csv")


if __name__ == "__main__":
    main()