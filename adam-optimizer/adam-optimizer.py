import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    m = m * beta1 + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_new = m / (1 - beta1 ** t)
    v_new = v / (1 - beta2 ** t)
    param = param - (lr * (m_new / (np.sqrt(v_new)+eps)))
    return (param, m, v)