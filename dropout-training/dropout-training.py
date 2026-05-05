import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    if rng is None:
        rng = np.random

    # Create mask: keep with probability (1 - p)
    mask = rng.random(x.shape) > p

    # Convert boolean mask to float and scale (inverted dropout)
    mask = mask.astype(x.dtype) / (1 - p)

    # Apply mask
    out = x * mask

    return out, mask