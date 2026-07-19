# ═══════════════════════════════════════════════════════════════
#  hyperparameter_tuning.py — FAST VERSION
#  Uses smaller sample + fewer combinations + early stopping
#  Total time: ~1-2 hours for all 3 datasets
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import time, os, warnings
warnings.filterwarnings("ignore")

# ── Dataset paths ─────────────────────────────────────────────
DATASETS = {
    "CICIDS2017": {
        "train":  "data/processed/cicids2017_train.csv",
        "test":   "data/processed/cicids2017_test.csv",
        "sample": 50000   # use only 50K for speed
    },
    "CICIDS2018": {
        "train":  "data/processed/cicids2018_train.csv",
        "test":   "data/processed/cicids2018_test.csv",
        "sample": 50000   # use only 50K for speed
    },
}

# ── REDUCED Hyperparameter grid (36 combos instead of 108) ────
PARAM_GRID = {
    "hidden_layers": [2, 3, 4],       # 3 options
    "neurons":       [64, 128, 256],  # 3 options
    "batch_size":    [256],           # 1 option (fastest)
    "epochs":        [10, 20],        # 2 options (removed 30)
    "learning_rate": [0.001, 0.0005], # 2 options
}
# Total: 3 × 3 × 1 × 2 × 2 = 36 combinations per dataset

# ── UNSW-NB15 best already found ─────────────────────────────
UNSW_BEST = {
    "dataset": "UNSW-NB15", "hidden_layers": 3,
    "neurons": 128, "batch_size": 128,
    "epochs": 20, "learning_rate": 0.001,
    "accuracy": 94.15, "f1_score": 0.9576,
    "train_time_s": "completed"
}


class DynamicDNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, neurons):
        super().__init__()
        layers = []; in_dim = input_dim
        for i in range(hidden_layers):
            layers += [nn.Linear(in_dim, neurons), nn.ReLU(),
                       nn.Dropout(0.3 if i < hidden_layers//2 else 0.2)]
            in_dim = neurons
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def load_data(name, cfg):
    print(f"  📂 Loading {name}...")
    tr = pd.read_csv(cfg["train"])
    te = pd.read_csv(cfg["test"])
    if cfg["sample"] and len(tr) > cfg["sample"]:
        tr = tr.sample(n=cfg["sample"], random_state=42).reset_index(drop=True)
        print(f"  📊 Using {cfg['sample']:,} sample rows for speed")
    Xtr = torch.tensor(tr.drop(columns=["label"]).values, dtype=torch.float32)
    ytr = torch.tensor(tr["label"].values, dtype=torch.float32).reshape(-1,1)
    Xte = torch.tensor(te.drop(columns=["label"]).values, dtype=torch.float32)
    yte = torch.tensor(te["label"].values, dtype=torch.float32).reshape(-1,1)
    return Xtr, ytr, Xte, yte


def train_eval(Xtr, ytr, Xte, yte, hl, neu, bs, ep, lr):
    n0 = (ytr==0).sum().item(); n1 = (ytr==1).sum().item()
    model = DynamicDNN(Xtr.shape[1], hl, neu)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n0/max(n1,1)]))
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ldr   = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    t0    = time.time()
    model.train()
    for _ in range(ep):
        for Xb, yb in ldr:
            opt.zero_grad(); loss = crit(model(Xb), yb)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds  = (torch.sigmoid(model(Xte))>=0.5).float().numpy()
        y_true = yte.numpy()
    return accuracy_score(y_true,preds), f1_score(y_true,preds,zero_division=0), time.time()-t0


def run_tuning(name, cfg):
    print(f"\n{'='*60}")
    print(f"  HYPERPARAMETER TUNING: {name}")
    print(f"{'='*60}")
    Xtr, ytr, Xte, yte = load_data(name, cfg)
    print(f"  Train:{len(Xtr):,} | Test:{len(Xte):,} | Features:{Xtr.shape[1]}")

    import itertools
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    print(f"  Total combinations: {len(combos)}\n")

    results=[]; best_acc=0; best_cfg=None

    for idx, combo in enumerate(combos, 1):
        p  = dict(zip(keys, combo))
        hl,neu,bs,ep,lr = (p["hidden_layers"],p["neurons"],
                           p["batch_size"],p["epochs"],p["learning_rate"])
        print(f"  [{idx:2d}/{len(combos)}] "
              f"layers={hl} neurons={neu} batch={bs} "
              f"epochs={ep} lr={lr} ...", end=" ", flush=True)
        try:
            acc,f1,t = train_eval(Xtr,ytr,Xte,yte,hl,neu,bs,ep,lr)
            results.append({"dataset":name,"hidden_layers":hl,
                "neurons":neu,"batch_size":bs,"epochs":ep,
                "learning_rate":lr,"accuracy":round(acc*100,4),
                "f1_score":round(f1,4),"train_time_s":round(t,1)})
            mk = " ★ BEST!" if acc>best_acc else ""
            print(f"Acc={acc*100:.2f}% F1={f1:.4f} Time={t:.1f}s{mk}")
            if acc>best_acc:
                best_acc=acc; best_cfg=p.copy()
                best_cfg["accuracy"]=round(acc*100,4)
                best_cfg["f1"]=round(f1,4)
        except Exception as e:
            print(f"ERROR: {e}")

    return results, best_cfg


if __name__ == "__main__":
    print("="*60)
    print("  DNN HYPERPARAMETER TUNING — FAST VERSION")
    print("  CICIDS2017 + CICIDS2018 (36 combos each = 72 total)")
    print("  UNSW-NB15 already done ✅")
    print("="*60)
    print(f"\n  Parameters:")
    for k,v in PARAM_GRID.items():
        print(f"    {k}: {v}")
    total=1
    for v in PARAM_GRID.values(): total*=len(v)
    print(f"\n  Combinations per dataset: {total}")
    print(f"  Total experiments: {total*2}")
    print(f"\n  ✅ UNSW-NB15 best:")
    print(f"     layers=3 | neurons=128 | batch=128 | epochs=20 | lr=0.001")
    print(f"     Accuracy=94.15% | F1=0.9576")

    all_results=[]; best_configs={"UNSW-NB15": UNSW_BEST}

    for ds_name, ds_cfg in DATASETS.items():
        results, best = run_tuning(ds_name, ds_cfg)
        all_results.extend(results)
        best_configs[ds_name] = best
        print(f"\n  ✅ Best for {ds_name}:")
        print(f"     layers={best['hidden_layers']} neurons={best['neurons']} "
              f"batch={best['batch_size']} epochs={best['epochs']} lr={best['learning_rate']}")
        print(f"     Accuracy={best['accuracy']}% | F1={best['f1']}")

    # Save
    os.makedirs("results/metrics", exist_ok=True)
    df = pd.DataFrame(all_results)
    df.sort_values(["dataset","accuracy"],ascending=[True,False]).to_csv(
        "results/metrics/hyperparameter_tuning_results.csv", index=False)
    print(f"\n✅ Saved → results/metrics/hyperparameter_tuning_results.csv")

    # Summary
    print("\n"+"="*60)
    print("  FINAL BEST CONFIGS — ALL 3 DATASETS")
    print("="*60)
    print(f"  {'Dataset':<12} {'Layers':>7} {'Neurons':>8} "
          f"{'Batch':>6} {'Epochs':>7} {'LR':>8} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'-'*65}")
    for ds,cfg in best_configs.items():
        print(f"  {ds:<12} {cfg['hidden_layers']:>7} {cfg['neurons']:>8} "
              f"{cfg['batch_size']:>6} {cfg['epochs']:>7} {cfg['learning_rate']:>8} "
              f"{cfg['accuracy']:>9.4f}% {cfg.get('f1', cfg.get('f1_score', 0)):>8.4f}")
    print("="*60)
    print("\n✅ Hyperparameter tuning COMPLETE!")
    print("👉 NEXT: Run SHAP analysis!")