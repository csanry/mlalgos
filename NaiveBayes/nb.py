import numpy as np

###################################
#STOPPED HERE
###################################


# Bayes Thm
# P(A|B) = P(B|A) * P(A) / P(B)

# In our case
# P(y|X) = P(X|y) * P(y) / P(X)
# with feature vector X = (x_1, x_2, x_n)...
# Naive Bayes assumes that all features are mutually independent

# Using chain rule;
# P(y|X) = P(x_1|y) * P(x_2|y) * ... * P(x_n|y) * P(y) / P(X)

# y = argmax_y P(y|X) -> argmax_y P(x_1|y) * P(x_2|y) * ... * P(x_n|y) * P(y)

# log all functions -> log(y) = argmax_y

class NaiveBayes:

    def fit(self, X, y):
        # X and y are np.arrays
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            # priors represents the frequency: how often the class is occurring
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        # /posterior_probs/ -> y = argmax_y log(P(x_1|y) + log(P(x_2|y) + ... + log(P(x_n|y) + log(P(y))
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            # calculate class conditionals
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numer = np.exp(- (x-mean)**2 / (2 * var))
        denom = np.sqrt(2 * np.pi * var)
        return numer / denom

    def score(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
