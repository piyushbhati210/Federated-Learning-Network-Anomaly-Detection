# ═══════════════════════════════════════════════════════════════
#  federated_rf_cicids.py
#  Federated Random Forest (Tree Aggregation)
#  for CICIDS2017 and CICIDS2018
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import pickle
import os
import copy
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    choices=["cicids2017", "cicids2018"])
args    = parser.parse_args()
DATASET = args.dataset

CLIENT_DATA_DIR  = f"../data/clients_{DATASET}"
TEST_FILE        = f"../data/processed/{DATASET}_test.csv"
LABEL_COL        = "label"
NUM_CLIENTS      = 4
TREES_PER_CLIENT = 25
OUTPUT_MODEL     = f"../models/global/fedavg_rf_{DATASET}.pkl"
OUTPUT_METRICS   = f"../results/metrics/federated_rf_{DATASET}_metrics.csv"


def main():
    print("=" * 60)
    print(f"  FEDERATED RANDOM FOREST — Tree Aggregation")
    print(f"  Dataset: {DATASET.upper()}")
    print("=" * 60)

    client_forests = []

    # ── Step 1: Each client trains local RF ───────────────────
    for cid in range(1, NUM_CLIENTS + 1):
        path = os.path.join(CLIENT_DATA_DIR,
                            f"client_{cid}_train.csv")
        df   = pd.read_csv(path)
        X    = df.drop(columns=[LABEL_COL])
        y    = df[LABEL_COL]

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

        local_preds = local_rf.predict(X)
        local_acc   = accuracy_score(y, local_preds)
        print(f"  ✅ Client {cid} done — "
              f"Local accuracy: {local_acc*100:.2f}%")

    # ── Step 2: Server aggregates all trees ───────────────────
    print("\n🌐 Server: Combining all client forests...")
    global_rf = copy.deepcopy(client_forests[0])
    all_trees = []
    for rf in client_forests:
        all_trees.extend(rf.estimators_)

    global_rf.estimators_  = all_trees
    global_rf.n_estimators = len(all_trees)
    print(f"  ✅ Global forest: {len(all_trees)} total trees "
          f"({NUM_CLIENTS} clients × {TREES_PER_CLIENT} each)")

    # ── Step 3: Evaluate on test set ──────────────────────────
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
    print(f"  📊 FEDERATED RF — {DATASET.upper()} RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, preds,
          target_names=['Normal', 'Attack']))
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_MODEL),   exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_METRICS), exist_ok=True)

    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(global_rf, f)
    print(f"\n✅ Model saved → {OUTPUT_MODEL}")

    pd.DataFrame([{
        "dataset":          DATASET,
        "model":            "Federated Random Forest",
        "num_clients":      NUM_CLIENTS,
        "trees_per_client": TREES_PER_CLIENT,
        "total_trees":      len(all_trees),
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1
    }]).to_csv(OUTPUT_METRICS, index=False)
    print(f"✅ Metrics saved → {OUTPUT_METRICS}")


if __name__ == "__main__":
    main()