import numpy as np

def selu(x, lam=1.0507009873554805,
         alpha=1.6732632423543772):
    """
    Apply SELU activation element-wise.
    Returns a list of floats rounded to 4 decimal places.
    """
    x = np.array(x, dtype=float)

    y = np.where(
        x > 0,
        lam * x,
        lam * alpha * (np.exp(x) - 1)
    )

    return np.round(y, 4).tolist()