import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=float)

    if x.ndim == 2:  # (N, D)
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + eps)
        y = gamma * x_norm + beta

    elif x.ndim == 4:  # (N, C, H, W)
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        x_norm = (x - mean) / np.sqrt(var + eps)

        gamma = np.asarray(gamma).reshape(1, -1, 1, 1)
        beta = np.asarray(beta).reshape(1, -1, 1, 1)

        y = gamma * x_norm + beta

    else:
        raise ValueError("Input must have shape (N,D) or (N,C,H,W)")

    return y