import numpy as np
from collections import Counter

def _euclidean_distance(x1, x2):
    """
    Computes the Euclidean distance of 2 points in Euclidean space, represented as a vector array.

    Parameters
    ----------
    x1 : Point 1, an array of coordinates.
    x2 : Point 2, an array of coordinates.

    Returns
    -------
    A distance metric.
    """
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    """
    Instantiates a KNN object to compute the KNN algorithm on data.
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Given new unseen data, predict class labels based on train dataset.

        Parameters
        ----------
        X : Array of unseen data observations

        Returns
        -------
        Array of predicted labels.
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        """
        Helper function for KNN.predict().

        Parameters
        ----------
        x : Single data observation in test data

        Returns
        -------
        Predicted label value.
        """
        # compute distances between the data point and all other points in self.X_train
        dist_from_all_pts = [_euclidean_distance(x, x_train_pt) for x_train_pt in self.X_train]

        # get k-nearest neighbors and their labels
        k_indices = np.argsort(dist_from_all_pts)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        # compute majority vote, return most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self, y_true, y_pred):
        """
        Compute the accuracy of predictions.

        Parameters
        ----------
        y_true : Array of true labels
        y_pred : Array of predicted labels

        Returns
        -------
        Float value representing the accuracy of predictions
        """
        return np.sum(y_pred == y_true) / len(y_pred)