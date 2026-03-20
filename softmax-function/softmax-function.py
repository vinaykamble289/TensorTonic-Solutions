import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x_arr = np.array(x)

    if x_arr.ndim == 1:
        x_shifted = x_arr - np.max(x_arr)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)

    elif x_arr.ndim == 2:
        x_shifted = x_arr - np.max(x_arr, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)