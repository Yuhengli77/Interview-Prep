import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class LogisticRegression:
    def __init__(self, input_dim):
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = np.zeros((1,))

    def forward(self, X):
        """
        return logits, shape (B, 1)
        """
        return X @ self.W + self.b

    def predict_proba(self, X):
        logits = self.forward(X)
        return sigmoid(logits)

    def binary_cross_entropy_loss(self, X, y_true, eps=1e-12):
        """
        y_true: shape (B, 1), values are 0 or 1
        """
        probs = self.predict_proba(X)
        probs = np.clip(probs, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
        return loss

    def gradients(self, X, y_true):
        """
        gradient of BCE loss
        y_true: (B, 1)
        """
        B = X.shape[0]
        logits = self.forward(X)          # (B, 1)
        probs = sigmoid(logits)           # (B, 1)

        # sigmoid + BCE 的经典结果
        dlogits = (probs - y_true) / B    # (B, 1)

        dW = X.T @ dlogits                # (D, 1)
        db = np.sum(dlogits, axis=0)      # (1,)

        return dW, db
    
model = LogisticRegression(input_dim=4)

X = np.random.randn(8, 4)
y = np.random.randint(0, 2, size=(8, 1))

loss = model.binary_cross_entropy_loss(X, y)
dW, db = model.gradients(X, y)

lr = 0.1
model.W -= lr * dW
model.b -= lr * db
