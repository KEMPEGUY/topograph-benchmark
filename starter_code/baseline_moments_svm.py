# ============================================================
# starter_code/baseline_moments_svm.py  (COMPETITION FORMAT)
# - Uses OFFICIAL split files: train_ids.npy / val_ids.npy / test_ids.npy
# - Trains on moments (train) + labels_train.npy
# - Validates on labels_val.npy if provided
# - Writes submissions/predictions.csv for OFFICIAL test_ids
# - Deterministic (seeded)
# - Prints "Wrote ..." ONLY ONCE (from save_predictions_csv)
# ============================================================

import random
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Robust import: works for both
#   python starter_code/baseline_moments_svm.py
#   python -m starter_code.baseline_moments_svm
try:
    from starter_code.utils_submit import save_predictions_csv
except ModuleNotFoundError:
    from utils_submit import save_predictions_csv


def main():
    # ----------------------------
    # Determinism
    # ----------------------------
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # ----------------------------
    # Paths
    # ----------------------------
    ROOT = Path(__file__).resolve().parents[1]     # repo root
    pub = ROOT / "data" / "public"
    out_path = ROOT / "submissions" / "predictions.csv"

    # ----------------------------
    # Load public features + official split ids
    # ----------------------------
    X = np.load(pub / "moments_all.npy").astype(np.float32)

    idx_train = np.load(pub / "train_ids.npy").astype(int)
    idx_val   = np.load(pub / "val_ids.npy").astype(int)
    idx_test  = np.load(pub / "test_ids.npy").astype(int)

    # Public labels
    y_train = np.load(pub / "labels_train.npy").astype(int)

    labels_val_path = pub / "labels_val.npy"
    y_val = np.load(labels_val_path).astype(int) if labels_val_path.exists() else None

    # ----------------------------
    # Train
    # ----------------------------
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale")
    )
    model.fit(X[idx_train], y_train)

    # ----------------------------
    # Validate (only if labels_val.npy exists)
    # ----------------------------
    print("\n=== Baseline: Moments-SVM (Competition Format) ===")
    if y_val is not None:
        pred_val = model.predict(X[idx_val])
        acc = accuracy_score(y_val, pred_val)
        mf1 = f1_score(y_val, pred_val, average="macro")
        print(f"{'Split':10s}  {'Acc':>8s}  {'Macro-F1':>10s}")
        print(f"{'Val':10s}  {acc:8.4f}  {mf1:10.4f}")
    else:
        print("[Info] labels_val.npy not found -> skipping validation metrics.")

    print(f"[Info] train_ids={len(idx_train)}  test_ids={len(idx_test)}  moments_dim={X.shape[1]}")

    # ----------------------------
    # Predict official test split + save (ONE print comes from utils)
    # ----------------------------
    pred_test = model.predict(X[idx_test])
    save_predictions_csv(ids=idx_test, y_pred=pred_test, out_path=str(out_path))


if __name__ == "__main__":
    main()

