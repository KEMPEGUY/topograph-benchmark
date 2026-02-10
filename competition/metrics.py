from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

def _to_label_preds(y_pred, threshold: float = 0.5):
    """
    Convert predictions to integer class labels.
    Accepts probabilities/logits or already-integer labels.
    """
    y_pred = np.asarray(y_pred)

    # If already integer-like (0/1), keep
    if np.issubdtype(y_pred.dtype, np.integer):
        return y_pred.astype(int)

    # Otherwise threshold floats
    return (y_pred >= threshold).astype(int)

def macro_f1(y_true, y_pred, threshold: float = 0.5):
    """
    Macro-F1 for (binary) classification.
    y_pred can be probabilities or labels.
    """
    y_true = np.asarray(y_true).astype(int)
    y_hat = _to_label_preds(y_pred, threshold=threshold)
    return float(f1_score(y_true, y_hat, average="macro"))

def binary_auc(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))

