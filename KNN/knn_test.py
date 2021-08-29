from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

from knn import KNN

knn = KNN(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = knn.score(y_test, y_pred)

print('Accuracy of KNN predictions:', round(accuracy, 2))