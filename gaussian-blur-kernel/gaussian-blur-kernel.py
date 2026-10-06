import math
import numpy as np

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    center = size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    sum = g.sum()
    return (g / sum).tolist()