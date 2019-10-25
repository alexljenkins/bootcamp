# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:22:59 2019
@author: alexl

PCA - Dimensionality Reduction
Finds the most important property/meta-feature in the data
This could be a linear combination of the features!!! (new compared to alternatives)

M is the principal components
M = 1 would find the single most important set of combination features
M = #of features. then features are ranked

Alternatives:
    - Recursive Feature Elimination
    - Lasso
    - P-Value outputs and manual removal
    - Stacastic PCA (for bigger datasets)
    - Kernal PCA (non-linear transformations)

Workflow:
    1. Develop Features
    2. de-mean and scale
    3. fit PCA
    4. train model on transformed data
    5. optimize # of components
"""



import pandas as pd
df = pd.read_csv('Data/titanictrain.csv', index_col=0)
df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Sex'] = (df['Sex'] == 'female').astype(int)

df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1)
del df['Embarked']
df.dropna(inplace=True)
y = df['Survived']
del df['Survived']
df.shape


#Data MUST HAVE a mean of 0 for all columns
X = df - df.mean()

from sklearn.decomposition import PCA
pca = PCA(n_components = 1) #reduce the dimention of each passenger to a single value

pca = PCA(n_components = 9)
A = pca.fit_transform(X)

A.shape
A[:3] #principal components on each passenger


#shows the varation in the data that's most important
from matplotlib import pyplot as plt
plt.bar(range(9),pca.components_[0])
plt.xticks(range(9),df.columns)
plt.show()


from sklearn.linear_model import LogisticRegression
m = LogisticRegression(C = 10000000)
m.fit(A, y)

print(m.score(A, y))

#hows how much information in is the data
print(pca.explained_variance_ratio_.round(3))

plt.bar(range(9),pca.explained_variance_ratio_)