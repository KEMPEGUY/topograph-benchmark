import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from competition.metrics import macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, required=True, help="Path to predictions.csv")
    args = parser.parse_args()

    root = Path(".").resolve()
    pub = root / "data" / "public"
    priv = root / "data" / "private"

    test_ids_path = pub / "test_ids.npy"
    labels_all_path = priv / "labels_all.npy"  # hidden, provided in CI via GitHub Secret

    if not test_ids_path.exists():
        raise FileNotFoundError(f"Missing public file: {test_ids_path}")
    if not labels_all_path.exists():
        raise FileNotFoundError(
            f"Missing hidden labels: {labels_all_path}. "
            f"In CI this file must be created from a GitHub Secret."
        )

    test_ids = np.load(test_ids_path).astype(int)
    labels_all = np.load(labels_all_path).astype(int)

    # ---- load submission
    sub_path = Path(args.submission)
    df = pd.read_csv(sub_path)

    if "id" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("Submission must have columns: id,y_pred")

    # enforce exact set match
    sub_ids = df["id"].to_numpy().astype(int)
    if len(sub_ids) != len(test_ids):
        raise ValueError(f"Submission has {len(sub_ids)} rows, expected {len(test_ids)}")

    if len(np.unique(sub_ids)) != len(sub_ids):
        raise ValueError("Submission has duplicate ids")

    # match order by sorting both by id
    df = df.sort_values("id").reset_index(drop=True)
    test_ids_sorted = np.sort(test_ids)

    if not np.array_equal(df["id"].to_numpy().astype(int), test_ids_sorted):
        raise ValueError("Submission ids do not match test_ids.npy exactly")

    y_pred = df["y_pred"].to_numpy().astype(int)
    y_true = labels_all[test_ids_sorted]

    score = macro_f1(y_true, y_pred)
    print(f"macro_f1: {score:.6f}")

   


if __name__ == "__main__":
    main()

