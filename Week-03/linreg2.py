
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipline import make_pipeline
from sklearn.linear_selection import train_test_split
pipeline = make_pipeline(
        PolynomialFeatures(9),
        MinMaxScaler(),
        Ridge(alpha=0.0)
        )


np.random.seed(42)
x = np.linspace(0, 20, 21)
y = 5 * x + 2 + np.random.normal(0.0, 20.0, 21)

# Hint: if you get a shape error from scikit, try:
X = x.reshape(21, 1)

poly = PolynomialFeatures(30)
Xpoly = poly.fit_transform(X)
Xscaled = MinMaxScaler().fit_transform(Xpoly)

Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, y, random_state=42)

m = Ridge(alpha=0.1)
m.fit(Xtrain, ytrain)
m.score(Xtrain, ytrain)
m.score(Xtest, ytest)
ypred = m.predict(Xscaled)

plt.bar(range(31), m.coef_)

plt.plot(Xtrain[:,1], ytrain, 'bo')
plt.plot(Xtest[:,1], ytest, 'kx')
plt.plot(Xscaled[:,1], ypred, 'r-')
plt.axis([0.0, 1.0, 0.0, 140.0])


plt.plot(Xtrain[:,1], ytrain, 'bo')
plt.plot(Xtest[:,1], ytest, 'kx')
plt.plot(x, y, 'bo')
plt.plot(x, ypred, 'r-')
plt.axis([2.0, 20.0, 20.0, 140.0])



"""
Extra, add:

* plot
* train-test-split
* scaling
* polynomials
* statsmodels
"""
