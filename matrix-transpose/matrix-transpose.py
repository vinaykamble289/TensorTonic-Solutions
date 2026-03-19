import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A_arr = np.array(A)
    rows, cols = A_arr.shape
    B = np.zeros((cols, rows)) 
    for i in range(A_arr.shape[0]):
        for j in range(A_arr.shape[1]):
            B[j][i] = A_arr[i][j]

    return B
        
