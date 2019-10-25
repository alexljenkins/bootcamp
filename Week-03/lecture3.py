# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:59:19 2019

@author: alexl
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


x = np.linspace(1,10,10)
y = 2* x + 3 + np.random.normal(0.0,5.0,10)
X = x.reshape(10,1)


p = PolynomialFeatures(3)
Xt = p.fit_transform(X)
xit = p.transform(xi.reshape(1000,1))

m = LinearRegression
m.fit(X,y)

ypred = m.predict(X)

plt.plot(x,y,'bo')
plt.plot(x,ypred,'r-')