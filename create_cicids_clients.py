# ═══════════════════════════════════════════════════════════════
#  create_cicids_clients.py
#  Split CICIDS2017 and CICIDS2018 training data into
#  4 FL client partitions each
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import os

def create_clients(train_path, dataset_name, num_clients=4):
    print(f"\n{'='*55}")
    print(f"  Creating {num_clients} client splits for {dataset_name}")
    print(f"{'='*55}")

    if not os.path.exists(train_path):
        print(f"❌ File not found: {train_path}")
        return

    print(f"📂 Loading {train_path}...")
    df = pd.read_csv(train_path)
    print(f"  ✅ Total training samples : {len(df):,}")
    print(f"  ✅ Features               : {df.shape[1]-1}")
    print(f"  ✅ Normal (0)             : {(df['label']==0).sum():,}")
    print(f"  ✅ Attack (1)             : {(df['label']==1).sum():,}")

    folder = f"data/clients_{dataset_name}"
    os.makedirs(folder, exist_ok=True)

    client_size = len(df) // num_clients
    print(f"\n  Splitting into {num_clients} clients "
          f"(~{client_size:,} samples each)...")

    for i in range(num_clients):
        start     = i * client_size
        end       = (i+1)*client_size if i < num_clients-1 else len(df)
        client_df = df.iloc[start:end].reset_index(drop=True)
        path      = f"{folder}/client_{i+1}_train.csv"
        client_df.to_csv(path, index=False)
        n0 = (client_df['label']==0).sum()
        n1 = (client_df['label']==1).sum()
        print(f"  ✅ Client {i+1}: {len(client_df):,} samples "
              f"(Normal={n0:,}, Attack={n1:,}) → {path}")

    print(f"\n  ✅ {dataset_name} client splits created in '{folder}/'")


if __name__ == "__main__":
    print("=" * 55)
    print("  CREATING FL CLIENT SPLITS")
    print("  CICIDS2017 + CICIDS2018")
    print("=" * 55)

    # CICIDS2017
    create_clients(
        train_path="data/processed/cicids2017_train.csv",
        dataset_name="cicids2017"
    )

    # CICIDS2018
    create_clients(
        train_path="data/processed/cicids2018_train.csv",
        dataset_name="cicids2018"
    )

    print("\n" + "=" * 55)
    print("  ✅ ALL CLIENT SPLITS CREATED!")
    print("  Folders created:")
    print("    data/clients_cicids2017/client_1_train.csv ... client_4")
    print("    data/clients_cicids2018/client_1_train.csv ... client_4")
    print("=" * 55)
    print("\n👉 NEXT: Run FL server and clients for each dataset!")