import numpy as np
import math

def novelty_score(recommendations: list, item_counts: list, n_users: int) -> float:
    item_counts = np.array(item_counts)
    sum = np.sum(-np.log2((item_counts / n_users)))
    return sum / len(recommendations)