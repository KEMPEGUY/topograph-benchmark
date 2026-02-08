import os
import numpy as np
from sklearn.model_selection import train_test_split

def repo_root():
    return os.path.dirname(os.path.dirname(__file__))

def load_public_arrays():
    root = repo_root()
    data_dir = os.path.join(root, "data", "public")

    X_all = np.load(os.path.join(data_dir, "moments_all.npy")).astype(np.float32)
    train_ids = np.load(os.path.join(data_dir, "train_ids.npy")).astype(np.int64)
    test_ids  = np.load(os.path.join(data_dir, "test_ids.npy")).astype(np.int64)
    y_train   = np.load(os.path.join(data_dir, "labels_train.npy")).astype(np.int64)

    assert len(train_ids) == len(y_train), "train_ids and labels_train length mismatch"
    return X_all, train_ids, test_ids, y_train

def make_val_split_from_train(train_ids, y_train, seed=0, val_ratio=0.2):
    # Split within TRAIN ONLY for reporting baselines
    idx = np.arange(len(train_ids))
    tr_local, val_local = train_test_split(
        idx, test_size=val_ratio, stratify=y_train, random_state=seed
    )
    train_idx = train_ids[tr_local]
    val_idx   = train_ids[val_local]
    return train_idx, val_idx

