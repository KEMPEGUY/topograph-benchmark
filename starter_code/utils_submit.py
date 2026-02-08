# ============================================================
# starter_code/utils_submit.py
# - Writes submissions/predictions.csv in the required format:
#     id,y_pred
# - Creates parent folder if missing
# - Prints ONE confirmation line
# ============================================================

import os
import numpy as np


def save_predictions_csv(ids, y_pred, out_path="submissions/predictions.csv"):
    """
    Save predictions in the official competition format.

    Parameters
    ----------
    ids : array-like of int
        The instance/graph IDs to predict for (must match test_ids.npy order/values).
    y_pred : array-like of int
        Predicted class labels.
    out_path : str
        Output CSV path, default: submissions/predictions.csv
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ids = np.asarray(ids).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(ids) != len(y_pred):
        raise ValueError(f"ids and y_pred must have same length: {len(ids)} vs {len(y_pred)}")

    with open(out_path, "w") as f:
        f.write("id,y_pred\n")
        for i, p in zip(ids, y_pred):
            f.write(f"{int(i)},{int(p)}\n")

    print(f"[OK] Wrote {out_path} with {len(ids)} rows.")

