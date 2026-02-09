import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred):
    """
    Compute macro F1 score.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(f1_score(y_true, y_pred, average="macro"))

