import numpy as np

def sigmoid(x):
    """
    Fully vectorized sigmoid function.
    Works for scalars, lists, and numpy arrays.
    """
    x = np.array(x)
    return 1 / (1 + np.exp(-x))