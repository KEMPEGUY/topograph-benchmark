from pathlib import Path
import numpy as np
import pandas as pd


def validate_submission_csv(submission_path: str) -> None:
    root = Path(".").resolve()
    pub = root / "data" / "public"
    test_ids = np.load(pub / "test_ids.npy").astype(int)
    test_ids_sorted = np.sort(test_ids)

    df = pd.read_csv(submission_path)

    # columns
    if "id" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("Submission must have columns: id,y_pred")

    # length
    if len(df) != len(test_ids):
        raise ValueError(f"Expected {len(test_ids)} rows, got {len(df)}")

    # types
    df["id"] = df["id"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)

    # duplicates
    if df["id"].duplicated().any():
        raise ValueError("Duplicate ids found in submission")

    # exact id set match
    df_sorted = df.sort_values("id").reset_index(drop=True)
    if not np.array_equal(df_sorted["id"].to_numpy().astype(int), test_ids_sorted):
        raise ValueError("Submission ids must match data/public/test_ids.npy exactly")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--submission", required=True, type=str)
    args = p.parse_args()

    validate_submission_csv(args.submission)
    print("[OK] submission is valid")

