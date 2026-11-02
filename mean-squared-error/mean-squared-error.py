import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return np.sum((y_pred-y_true) ** 2) / len(y_pred)