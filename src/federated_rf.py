"""
federated_rf.py
----------------------------------
Federated Random Forest for Network Traffic Anomaly Detection (UNSW-NB15)

WHY THIS IS DIFFERENT FROM FedAvg (used for LR / DNN):
Random Forest has no numeric weight vector to average across clients.
A tree is a sequence of if/else splits, not a set of tunable weights.
So instead of FedAvg, this script uses the standard federated approach
for tree ensembles: each client trains its OWN local Random Forest on
its own private data slice, and the server simply combines every
client's trained trees into one larger forest (a forest of forests).

This still satisfies the core Federated Learning goal — no client ever
shares its raw data, only its trained trees — while working correctly
with a non-gradient-based model like Random Forest.

Run from your FL_Anomaly_Detection folder:
    python src/federated_rf.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)

# ── Paths (using ../ because this file is inside src/ folder) ──
CLIENT_DATA_DIR = "../data/clients"
TEST_FILE       = "../data/processed/test_cleaned.csv"
LABEL_COL       = "label"
NUM_CLIENTS     = 4
TREES_PER_CLIENT = 25
OUTPUT_MODEL    = "../models/global/fedavg_random_forest.pkl"
OUTPUT_METRICS  = "../results/metrics/federated_rf_metrics.csv"


def load_client_data(client_id):
    path = os.path.join(CLIENT_DATA_DIR,
                        f"client_{client_id}_train.csv")
    df   = pd.read_csv(path)
    X    = df.drop(columns=[LABEL_COL])
    y    = df[LABEL_COL]
    return X, y


def main():
    print("=" * 60)
    print("  FEDERATED RANDOM FOREST — Tree Aggregation")
    print("  Federated Learning - Network Traffic Anomaly Detection")
    print("=" * 60)

    # ── Step 1: Each client trains local Random Forest ─────────
    client_forests = []
    for cid in range(1, NUM_CLIENTS + 1):
        X, y = load_client_data(cid)
        print(f"\n🔵 Client {cid}: Training local RF "
              f"({len(X):,} samples, {TREES_PER_CLIENT} trees)...")

        local_rf = RandomForestClassifier(
            n_estimators=TREES_PER_CLIENT,
            random_state=42 + cid,
            n_jobs=-1,
            class_weight='balanced'
        )
        local_rf.fit(X, y)
        client_forests.append(local_rf)

        # Quick local accuracy check
        local_preds = local_rf.predict(X)
        local_acc   = accuracy_score(y, local_preds)
        print(f"  ✅ Client {cid} done — "
              f"Local accuracy: {local_acc*100:.2f}%")

    # ── Step 2: Server aggregates all trees ────────────────────
    print("\n🌐 Server: Combining all client forests...")
    global_rf = copy.deepcopy(client_forests[0])
    all_estimators = []
    for rf in client_forests:
        all_estimators.extend(rf.estimators_)

    global_rf.estimators_  = all_estimators
    global_rf.n_estimators = len(all_estimators)
    print(f"  ✅ Global forest: {len(all_estimators)} total trees "
          f"({NUM_CLIENTS} clients × {TREES_PER_CLIENT} trees each)")

    # ── Step 3: Evaluate on shared test set ────────────────────
    print("\n📂 Loading test set...")
    test_df = pd.read_csv(TEST_FILE)
    X_test  = test_df.drop(columns=[LABEL_COL])
    y_test  = test_df[LABEL_COL]
    print(f"  ✅ Test samples: {len(X_test):,}")

    print("\n🔎 Evaluating Global Federated Random Forest...")
    preds = global_rf.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)

    print("\n" + "=" * 60)
    print("  📊 FEDERATED RANDOM FOREST — RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print(f"\nClassification Report:")
    print(classification_report(y_test, preds,
          target_names=['Normal', 'Attack']))

    # ── Comparison with centralized RF ─────────────────────────
    print("=" * 60)
    print("  COMPARISON WITH CENTRALIZED MODELS (UNSW-NB15)")
    print("=" * 60)
    print(f"  {'Model':<40} {'Accuracy':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Centralized Logistic Regression':<40} {'93.23%':>10}")
    print(f"  {'Centralized Random Forest':<40} {'96.07%':>10}")
    print(f"  {'Centralized DNN':<40} {'93.35%':>10}")
    print(f"  {'Federated LR (FedAvg, 4 clients)':<40} {'84.33%':>10}")
    print(f"  {'Federated DNN (FedAvg, 4 clients)':<40} {'93.91%':>10}")
    print(f"  {'Federated RF (Tree Aggregation)':<40} "
          f"{acc*100:>9.2f}%  ← NEW")
    print("=" * 60)

    # ── Save model ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_MODEL),   exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_METRICS), exist_ok=True)

    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(global_rf, f)
    print(f"\n✅ Global model saved → {OUTPUT_MODEL}")

    pd.DataFrame([{
        "model":            "Federated Random Forest",
        "num_clients":      NUM_CLIENTS,
        "trees_per_client": TREES_PER_CLIENT,
        "total_trees":      len(all_estimators),
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1
    }]).to_csv(OUTPUT_METRICS, index=False)
    print(f"✅ Metrics saved  → {OUTPUT_METRICS}")
    print("\n👉 Update federated_results in evaluate_compare.py "
          f"with RF federated accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()