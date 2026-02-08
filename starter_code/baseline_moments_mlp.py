# ============================================================
# starter_code/baseline_moments_mlp.py  (COMPETITION FORMAT)
# - Uses OFFICIAL split files: train_ids.npy / val_ids.npy / test_ids.npy
# - Trains on moments (train) + labels_train.npy
# - Validates on labels_val.npy if provided
# - Writes submissions/predictions.csv for OFFICIAL test_ids
# - Deterministic (seeded) via random_state + numpy/random seeds
# ============================================================

import random
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from starter_code.utils_submit import save_predictions_csv
except ModuleNotFoundError:
    from utils_submit import save_predictions_csv


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    ROOT = Path(__file__).resolve().parents[1]
    pub = ROOT / "data" / "public"
    out_path = ROOT / "submissions" / "predictions.csv"

    # Public features
    X = np.load(pub / "moments_all.npy").astype(np.float32)

    # Official split ids
    idx_train = np.load(pub / "train_ids.npy").astype(int)
    idx_val   = np.load(pub / "val_ids.npy").astype(int)
    idx_test  = np.load(pub / "test_ids.npy").astype(int)

    # Public labels
    y_train = np.load(pub / "labels_train.npy").astype(int)
    labels_val_path = pub / "labels_val.npy"
    y_val = np.load(labels_val_path).astype(int) if labels_val_path.exists() else None

    # Model: standardized MLP
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=1000,
            random_state=seed
        )
    )

    model.fit(X[idx_train], y_train)

    print("\n=== Baseline: Moments-MLP (Competition Format) ===")
    if y_val is not None:
        pred_val = model.predict(X[idx_val])
        acc = accuracy_score(y_val, pred_val)
        mf1 = f1_score(y_val, pred_val, average="macro")
        print(f"{'Split':10s}  {'Acc':>8s}  {'Macro-F1':>10s}")
        print(f"{'Val':10s}  {acc:8.4f}  {mf1:10.4f}")
    else:
        print("[Info] labels_val.npy not found -> skipping validation metrics.")

    # Predict official test + save
    pred_test = model.predict(X[idx_test])
    save_predictions_csv(ids=idx_test, y_pred=pred_test, out_path=str(out_path))


if __name__ == "__main__":
    main()

