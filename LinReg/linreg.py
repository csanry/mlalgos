import numpy as np

## Math
# y_hat = weights * X + b
# Cost function >
# MSE = J(w, b) = 1/N sum_of (y_i - (w * x_i + b)) ** 2

# Update rules > gradient descent
# new_w = w - learn_rate * deriv(w)
# new_b = b - learn_rate * deriv(b)

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape

        # set initial weights and bias to 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # calculate the approximation by updating params using gradient descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            # partial derivatives of each parameter
            dw = (2/n_samples) * np.dot(X.T, (y_pred-y))
            db = (2/n_samples) * np.sum(y_pred-y)

            # weights and bias will converge by end of loop
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
