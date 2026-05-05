import numpy as np

def conv2d(x, W, b):
    """
    2D Convolution (channel-first)

    x: (N, C_in, H, W)
    W: (C_out, C_in, KH, KW)
    b: (C_out,)

    returns:
    y: (N, C_out, H_out, W_out)
    """

    x = np.array(x)
    W = np.array(W)
    b = np.array(b)

    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape

    # Output dimensions
    H_out = H - KH + 1
    W_out = W_in - KW + 1

    # Initialize output
    y = np.zeros((N, C_out, H_out, W_out))

    # Convolution
    for n in range(N):                  # over batch
        for c_out in range(C_out):      # over output channels (filters)
            for i in range(H_out):      # height
                for j in range(W_out):  # width
                    val = 0.0
                    for c_in in range(C_in):  # sum over input channels
                        region = x[n, c_in, i:i+KH, j:j+KW]
                        val += np.sum(region * W[c_out, c_in])
                    
                    y[n, c_out, i, j] = val + b[c_out]

    return y