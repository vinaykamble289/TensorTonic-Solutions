import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.array(x)

    den = np.exp(x) + np.exp(-x)
    nem = np.exp(x) - np.exp(-x)
    return nem / den