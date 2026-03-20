import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    Works for 1D or multi-dimensional X.
    """
    X = np.array(X)
    y = np.array(y)

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    # Initialize weights
    w = np.zeros(n_features)
    b = 0

    for _ in range(steps):
        Z = np.dot(X, w) + b
        y_pred = _sigmoid(Z)

        # Gradients
        dw = np.dot(X.T, (y_pred - y)) / n_samples
        db = np.sum(y_pred - y) / n_samples

        # Update
        w -= lr * dw
        b -= lr * db

    return w, b