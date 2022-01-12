from sklearn.model_selection import train_test_split
from sklearn import datasets
from logreg import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# instantiate a LogisticRegression classifier
lr = LogisticRegression(lr=0.01)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = lr.score(y_test, y_pred)
print('Logistic Regression classification accuracy:', accuracy)