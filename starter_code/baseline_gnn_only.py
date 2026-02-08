# ============================================================
# starter_code/baseline_gnn_only.py  (COMPETITION FORMAT)
# - Uses OFFICIAL split files: train_ids.npy / val_ids.npy / test_ids.npy
# - Trains a simple GCN on ENZYMES graphs (PyG)
# - Validates if labels_val.npy exists
# - Writes submissions/predictions.csv for OFFICIAL test_ids
# - Deterministic training (seed + deterministic DataLoader shuffle)
# ============================================================

import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.metrics import accuracy_score, f1_score

try:
    from starter_code.utils_submit import save_predictions_csv
except ModuleNotFoundError:
    from utils_submit import save_predictions_csv


def infer_in_dim(dataset):
    for d in dataset:
        if d.x is not None:
            return d.x.shape[1]
    return 1


class GNNOnly(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64, num_classes=6):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.lin = nn.Linear(out_dim, num_classes)

    def forward(self, data):
        x = data.x
        if x is None:
            x = torch.ones((data.num_nodes, 1), device=data.edge_index.device)
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        g = global_mean_pool(x, data.batch)
        return self.lin(g)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out.argmax(dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        if hasattr(batch, "y") and batch.y is not None:
            y_true.extend(batch.y.cpu().numpy().tolist())
    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int)


def main():
    # ----------------------------
    # Determinism
    # ----------------------------
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------------
    # Paths + split files
    # ----------------------------
    ROOT = Path(__file__).resolve().parents[1]
    pub = ROOT / "data" / "public"
    out_path = ROOT / "submissions" / "predictions.csv"

    idx_train = np.load(pub / "train_ids.npy").astype(int)
    idx_val   = np.load(pub / "val_ids.npy").astype(int)
    idx_test  = np.load(pub / "test_ids.npy").astype(int)

    labels_val_path = pub / "labels_val.npy"
    y_val = np.load(labels_val_path).astype(int) if labels_val_path.exists() else None

    # ----------------------------
    # Load dataset (PyG ENZYMES)
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TUDataset(root=str(ROOT / "data"), name="ENZYMES")

    # If labels_val.npy exists, we attach it to val graphs for evaluation
    # Training uses dataset labels internally (standard supervised training)
    # (This is fine because participants train offline; the hidden test labels remain private.)
    if y_val is not None:
        for pos, i in enumerate(idx_val):
            dataset[i].y = torch.tensor(int(y_val[pos]), dtype=torch.long)

    # ----------------------------
    # Dataloaders (deterministic shuffle)
    # ----------------------------
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader([dataset[i] for i in idx_train], batch_size=32, shuffle=True, generator=g)
    val_loader   = DataLoader([dataset[i] for i in idx_val],   batch_size=32, shuffle=False)
    test_loader  = DataLoader([dataset[i] for i in idx_test],  batch_size=32, shuffle=False)

    # ----------------------------
    # Model
    # ----------------------------
    in_dim = infer_in_dim(dataset)
    num_classes = int(dataset[0].y.max().item()) + 1 if dataset[0].y is not None else 6
    # safer: compute from dataset:
    all_y = [int(d.y.item()) for d in dataset if d.y is not None]
    num_classes = int(max(all_y)) + 1 if len(all_y) else 6

    model = GNNOnly(in_dim=in_dim, hidden=64, out_dim=64, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    # ----------------------------
    # Train
    # ----------------------------
    for _ in range(30):
        train_epoch(model, train_loader, opt, crit, device)

    # ----------------------------
    # Validate (if labels_val.npy exists)
    # ----------------------------
    print("\n=== Baseline: GNN-only (Competition Format) ===")
    if y_val is not None:
        y_true, y_pred = predict(model, val_loader, device)
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro")
        print(f"{'Split':10s}  {'Acc':>8s}  {'Macro-F1':>10s}")
        print(f"{'Val':10s}  {acc:8.4f}  {mf1:10.4f}")
    else:
        print("[Info] labels_val.npy not found -> skipping validation metrics.")

    # ----------------------------
    # Predict test + save
    # ----------------------------
    _, y_pred_test = predict(model, test_loader, device)
    save_predictions_csv(ids=idx_test, y_pred=y_pred_test, out_path=str(out_path))


if __name__ == "__main__":
    main()

