"""
federated_rf.py
----------------------------------
Federated Random Forest for Network Traffic Anomaly Detection (UNSW-NB15)

WHY THIS IS DIFFERENT FROM FedAvg (used for LR / DNN):
Random Forest has no numeric weight vector to average across clients.
A tree is a sequence of if/else splits, not a set of tunable weights.
So instead of FedAvg, this script uses the standard federated approach
for tree ensembles: each client trains its OWN local Random Forest on
its own private data slice, and the "server" simply combines every
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---- EDIT THESE if your paths differ ----
CLIENT_DATA_DIR = "data/clients"       # expects client_1_train.csv ... client_4_train.csv
TEST_FILE = "data/processed/test_cleaned.csv"
LABEL_COL = "label"                     # change if your label column is named differently
NUM_CLIENTS = 4
TREES_PER_CLIENT = 25                   # each client trains this many trees locally
OUTPUT_MODEL = "models/global/fedavg_random_forest.pkl"
OUTPUT_METRICS = "results/metrics/federated_rf_metrics.csv"
# ------------------------------------------


def load_client_data(client_id):
    path = os.path.join(CLIENT_DATA_DIR, f"client_{client_id}_train.csv")
    df = pd.read_csv(path)
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]
    return X, y


def main():
    print("=" * 60)
    print("  FEDERATED RANDOM FOREST — Tree Aggregation across Clients")
    print("=" * 60)

    if not os.path.exists(TEST_FILE):
        print(f"❌ Test file not found: {TEST_FILE}")
        print("Edit TEST_FILE at the top of this script.")
        return

    client_forests = []

    # --- Step 1: Each client trains its own local Random Forest ---
    for cid in range(1, NUM_CLIENTS + 1):
        try:
            X, y = load_client_data(cid)
        except FileNotFoundError:
            print(f"❌ Could not find data/clients/client_{cid}_train.csv — "
                  f"edit CLIENT_DATA_DIR at the top of this script if your "
                  f"filenames differ.")
            return

        print(f"🔵 Client {cid}: training local Random Forest "
              f"({len(X):,} samples, {TREES_PER_CLIENT} trees)...")

        local_rf = RandomForestClassifier(
            n_estimators=TREES_PER_CLIENT,
            random_state=42 + cid,   # different seed per client for tree diversity
            n_jobs=-1
        )
        local_rf.fit(X, y)
        client_forests.append(local_rf)
        print(f"  ✅ Client {cid} done.")

    # --- Step 2: Server aggregates all clients' trees into one global forest ---
    print("\n🌐 Server: combining all client forests into one global forest...")
    global_rf = copy.deepcopy(client_forests[0])
    all_estimators = []
    for rf in client_forests:
        all_estimators.extend(rf.estimators_)

    global_rf.estimators_ = all_estimators
    global_rf.n_estimators = len(all_estimators)
    print(f"  ✅ Global forest assembled: {len(all_estimators)} total trees "
          f"({NUM_CLIENTS} clients × {TREES_PER_CLIENT} trees each)")

    # --- Step 3: Evaluate the combined federated forest on the test set ---
    print("\n📂 Loading test set for evaluation...")
    test_df = pd.read_csv(TEST_FILE)
    X_test = test_df.drop(columns=[LABEL_COL])
    y_test = test_df[LABEL_COL]

    print("🔎 Evaluating Federated Random Forest...")
    preds = global_rf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print("\n" + "=" * 60)
    print("  📊 FEDERATED RANDOM FOREST — RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}")
    print("=" * 60)

    # --- Step 4: Save model and metrics ---
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(global_rf, f)
    print(f"\n✅ Global federated model saved: {OUTPUT_MODEL}")

    os.makedirs(os.path.dirname(OUTPUT_METRICS), exist_ok=True)
    pd.DataFrame([{
        "model": "Federated Random Forest",
        "num_clients": NUM_CLIENTS,
        "trees_per_client": TREES_PER_CLIENT,
        "total_trees": len(all_estimators),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }]).to_csv(OUTPUT_METRICS, index=False)
    print(f"✅ Metrics saved: {OUTPUT_METRICS}")
    print("\n👉 NEXT: Add this result to your report's Federated Learning comparison table!")


if __name__ == "__main__":
    main()