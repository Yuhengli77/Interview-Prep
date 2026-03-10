import numpy as np

def softmax(x, axis=-1):
    """
    x: np.ndarray, shape can be (..., num_classes)
    axis: dimension to normalize over
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)   # 数值稳定
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy_loss(logits, y_true):
    """
    logits: np.ndarray of shape (B, C)
    y_true: np.ndarray of shape (B,), each value in [0, C-1]
    return: scalar, mean cross-entropy loss
    """
    # log-sum-exp trick
    max_logits = np.max(logits, axis=1, keepdims=True)          # (B, 1)
    shifted = logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))  # (B, 1)
    log_probs = shifted - log_sum_exp                           # (B, C)

    batch_size = logits.shape[0]
    correct_log_probs = log_probs[np.arange(batch_size), y_true]  # (B,)
    loss = -np.mean(correct_log_probs)
    return loss



def cross_entropy_from_probs(probs, y_true, eps=1e-12):
    """
    probs: shape (B, C), each row sums to 1
    y_true: shape (B,)
    """
    batch_size = probs.shape[0]
    correct_probs = probs[np.arange(batch_size), y_true]
    correct_probs = np.clip(correct_probs, eps, 1.0)  # 防止 log(0)
    loss = -np.mean(np.log(correct_probs))
    return loss


def cross_entropy_grad(logits, y_true):
    """
    gradient of mean softmax cross-entropy loss w.r.t. logits
    logits: (B, C)
    y_true: (B,)
    return: (B, C)
    """
    probs = softmax(logits, axis=1)
    batch_size = logits.shape[0]

    grad = probs.copy()
    grad[np.arange(batch_size), y_true] -= 1
    grad /= batch_size
    return grad