"""
evaluate_compare.py
----------------------------------
Generates final comparison visuals + a master metrics CSV for the report.

Produces:
  1. centralized_comparison.png   -> bar chart: LR/RF/DNN accuracy across
                                      UNSW-NB15, CICIDS2017, CICIDS2018
  2. federated_vs_centralized.png -> bar chart: centralized vs federated
                                      accuracy for LR/RF/DNN on UNSW-NB15
  3. master_metrics.csv           -> every accuracy/F1 number in one table

This script uses the metric values already confirmed from your training
runs (hard-coded below) rather than re-running any models, since you
already have all the numbers from console output. If you'd rather this
script load results dynamically from saved files, tell me and I'll
adjust it — but hard-coding avoids re-running slow training just to
make a chart.

Run from your FL_Anomaly_Detection folder:
    python src/evaluate_compare.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "results/plots"
METRICS_OUTPUT = "results/metrics/master_metrics.csv"

# ----------------------------------------------------------------
# 1. CENTRALIZED RESULTS — all 3 datasets
# ----------------------------------------------------------------
centralized_results = [
    {"dataset": "UNSW-NB15",  "model": "Logistic Regression", "accuracy": 93.23, "f1": 0.9522},
    {"dataset": "UNSW-NB15",  "model": "Random Forest",       "accuracy": 96.07, "f1": 0.9714},
    {"dataset": "UNSW-NB15",  "model": "DNN",                 "accuracy": 93.35, "f1": 0.9498},
    {"dataset": "CICIDS2017", "model": "Logistic Regression", "accuracy": 96.69, "f1": 0.8937},
    {"dataset": "CICIDS2017", "model": "Random Forest",       "accuracy": 99.90, "f1": 0.9968},
    {"dataset": "CICIDS2017", "model": "DNN",                 "accuracy": 98.40, "f1": 0.9473},
    {"dataset": "CICIDS2018", "model": "Logistic Regression", "accuracy": 99.89, "f1": 0.9962},
    {"dataset": "CICIDS2018", "model": "Random Forest",       "accuracy": 100.00, "f1": 1.0000},
    {"dataset": "CICIDS2018", "model": "DNN",                 "accuracy": 99.98, "f1": 0.9994},
]

# ----------------------------------------------------------------
# 2. FEDERATED RESULTS — UNSW-NB15 only (so far)
# ----------------------------------------------------------------
federated_results = [
    {"model": "Logistic Regression", "centralized": 93.23, "federated": 84.33},
    {"model": "Random Forest",       "centralized": 96.07, "federated": 95.58},
    {"model": "DNN",                 "centralized": 93.35, "federated": 93.91},
]


def plot_centralized_comparison(df):
    datasets = df["dataset"].unique()
    models = df["model"].unique()
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, model in enumerate(models):
        values = [df[(df.dataset == d) & (df.model == model)]["accuracy"].values[0] for d in datasets]
        ax.bar(x + i * width, values, width, label=model)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Centralized Model Accuracy Across Datasets")
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.set_ylim(80, 101)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "centralized_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_federated_vs_centralized(df):
    models = df["model"]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, df["centralized"], width, label="Centralized")
    ax.bar(x + width/2, df["federated"], width, label="Federated (FedAvg / Tree Aggregation)")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Federated vs Centralized Accuracy (UNSW-NB15)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(75, 101)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, model in enumerate(models):
        c_val = df.iloc[i]["centralized"]
        f_val = df.iloc[i]["federated"]
        ax.text(i - width/2, c_val + 0.5, f"{c_val:.2f}%", ha="center", fontsize=9)
        ax.text(i + width/2, f_val + 0.5, f"{f_val:.2f}%", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "federated_vs_centralized.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {path}")


def main():
    print("=" * 60)
    print("  GENERATING COMPARISON CHARTS + MASTER METRICS")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)

    centralized_df = pd.DataFrame(centralized_results)
    federated_df = pd.DataFrame(federated_results)

    print("\n📊 Plotting centralized comparison (3 datasets)...")
    plot_centralized_comparison(centralized_df)

    print("\n📊 Plotting federated vs centralized (UNSW-NB15)...")
    plot_federated_vs_centralized(federated_df)

    print("\n💾 Saving master metrics CSV...")
    centralized_df["type"] = "Centralized"
    fed_rows = []
    for _, row in federated_df.iterrows():
        fed_rows.append({
            "dataset": "UNSW-NB15", "model": row["model"], "accuracy": row["federated"],
            "f1": None, "type": "Federated"
        })
    master_df = pd.concat([centralized_df, pd.DataFrame(fed_rows)], ignore_index=True)
    master_df.to_csv(METRICS_OUTPUT, index=False)
    print(f"  ✅ Saved: {METRICS_OUTPUT}")

    print("\n" + "=" * 60)
    print("  ✅ DONE — charts and metrics ready for your report/slides!")
    print("=" * 60)


if __name__ == "__main__":
    main()