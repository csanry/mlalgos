from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from linreg import LinearRegression

# generate a random regression problem
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

linreg = LinearRegression(lr=0.01)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
mse_score = linreg.mse(y_test, y_pred)
print('MSE:', mse_score)

y_pred_line = linreg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
train_pts = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
test_pts = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()