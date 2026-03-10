import numpy as np

class LinearRegression:
    def __init__(self, input_dim):
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = np.zeros((1,))

    def forward(self, X):
        """
        X: (B, D)
        return: (B, 1)
        """
        return X @ self.W + self.b

    def mse_loss(self, y_pred, y_true):
        """
        y_pred: (B, 1)
        y_true: (B, 1)
        """
        return np.mean((y_pred - y_true) ** 2)

    def gradients(self, X, y_true):
        """
        return gradients of MSE loss w.r.t. W and b
        """
        B = X.shape[0]
        y_pred = self.forward(X)                  # (B, 1)
        error = y_pred - y_true                   # (B, 1)

        # dL/dy_pred for mean squared error
        # L = mean((y_pred - y_true)^2)
        dy_pred = 2.0 * error / B                 # (B, 1)

        dW = X.T @ dy_pred                        # (D, 1)
        db = np.sum(dy_pred, axis=0)              # (1,)

        return dW, db
    

model = LinearRegression(input_dim=3)

X = np.random.randn(5, 3)
y = np.random.randn(5, 1)

y_pred = model.forward(X)
loss = model.mse_loss(y_pred, y)
dW, db = model.gradients(X, y)

lr = 0.01
model.W -= lr * dW
model.b -= lr * db