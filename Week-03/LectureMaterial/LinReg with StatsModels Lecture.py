# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:18:45 2019

@author: alexl
"""

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(1, 10, 1000)
y = 5 * x + 2 + np.random.normal(0.0, 3.0, 1000)
X = x.reshape((1000,1))

X = sm.add_constant(X) #adds a extra column full of 1's for y values
#remove this will set intercept to (0,0)

model = sm.OLS(y, X) # Ordinary Least Squares

results = model.fit()

print(results.summary())

plt.plot(x,y,'bo')

#plots the confidence interval
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from matplotlib import pyplot as plt
prstd, iv_l, iv_u = wls_prediction_std(results)
plt.plot(x, y, 'o', label="data")
plt.plot(x, results.fittedvalues, 'r--.', label="OLS")
plt.plot(x, iv_u, 'r--')
plt.plot(x, iv_l, 'r--')

import statsmodel.formula.api as smf
#this is r style modeling of y = lottery, x = lit, wealth and region
mod = smf.ols(formula = 'Lottery ~ literacy + Wealth + Region', data=df)
results = mod.fit()


