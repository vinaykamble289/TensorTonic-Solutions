import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat): 
    """Convert 1D array to 2D and track if conversion happened""" 
    a = np.asarray(a, dtype=float) 
    if a.ndim == 1: 
        return a.reshape(1, feat), True 
    return a, False

def _tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    shape = np.asarray(h_prev).shape
    x, x_2d = _as2d(x, np.asarray(x).shape[-1])
    h_prev, h_2d = _as2d(h_prev, np.asarray(h_prev).shape[-1])
    zt = _sigmoid(((x @ params["Wz"])+(h_prev @ params["Uz"]) + params["bz"]))
    rt = _sigmoid((x @ params["Wr"] + h_prev @ params["Ur"] + params["br"]))
    _ht = _tanh((x @ params["Wh"]+(rt * h_prev) @ params["Uh"] + params["bh"]))
    ht = ((1-zt)*h_prev) + (zt* _ht)
    
    return ht.reshape(shape)