import os
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(__file__))  # repo root
DATA = os.path.join(ROOT, "data", "public")

def main(seed=0):
    y = np.load(os.path.join(DATA, "labels_all.npy")).astype(np.int64)
    idx_all = np.arange(len(y))

    train_ids, test_ids = train_test_split(
        idx_all, test_size=0.2, stratify=y, random_state=seed
    )

    os.makedirs(DATA, exist_ok=True)
    np.save(os.path.join(DATA, "train_ids.npy"), train_ids.astype(np.int64))
    np.save(os.path.join(DATA, "test_ids.npy"),  test_ids.astype(np.int64))

    # Only expose labels for train
    y_train = y[train_ids]
    np.save(os.path.join(DATA, "labels_train.npy"), y_train.astype(np.int64))

    print("Saved:")
    print(" - data/public/train_ids.npy :", train_ids.shape)
    print(" - data/public/test_ids.npy  :", test_ids.shape)
    print(" - data/public/labels_train.npy :", y_train.shape)

if __name__ == "__main__":
    main(seed=0)

