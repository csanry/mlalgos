

##
# Inputs -> Weights -> Net_inp -> Activation_function -> Output

# Linear Model
# f(w, b) = w^Tx + b

# Activation function
# Unit step function
# g(z) = 1 if z > 0, 0 otherwise

# Approximation
# y_hat = g(f(w, b)) = g(w^Tx + b)

# Perceptron update rule
# For each training sample x_i:
# w := w + delta_w
# delta_w := alpha * (y_i - y_hat_i) * x_i
# a: learning rate in [0, 1]

import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # change to 1s and 0s if not true
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # apply update rule
                single_linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(single_linear_output)

                update = self.lr * (y_[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        # apply linear fn
        linear_output = np.dot(X, self.weights) + self.bias
        # apply activation fn
        y_pred = self.activation_func(linear_output)
        return y_pred