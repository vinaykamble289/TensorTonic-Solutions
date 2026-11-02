import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    y_prob = 0
    for i in range(len(y_true)):
        y_prob += np.log(y_pred[i][y_true[i]])
    return -y_prob / len(y_true)