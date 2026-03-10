import numpy as np

def softmax(x, axis=-1):
    """
    x: np.ndarray, shape can be (..., num_classes)
    axis: dimension to normalize over
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)   # 数值稳定
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)