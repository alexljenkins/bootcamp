from keras.datasets import mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

from matplotlib import pyplot as plt

def draw_array(x):
    plt.figure(figsize=(12,7))
    for i in range(40):
        plt.subplot(5, 8, i+1)
        plt.imshow(x[i], cmap=plt.cm.Greys)
        plt.axis('off')

draw_array(xtrain)


x_train = xtrain.reshape(60000,28*28)

#unsupervised learning for dimensionality reduction
from sklearn.decomposition import PCA
# principle component
# only works on de-meaned and on the same scale

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

#reduce 784 dimensions down to 100
m = PCA(100)
m.fit(x_train)

# explore the shape
m.components_.shape

m.components_[0] #Eigenvector of the original feature space with the largest eigenvalue.
# this is the vector, that when projecting the original data onto it,
# retains the largest share of the overall variance.

# The first dimension can explain 5.6% of the variance of the data
m.explained_variance_ratio_[0]

# the 100 dimensions can explain 70% of the variance in the data
m.explained_variance_ratio_.sum()

# (60000, 784) * (784, 100) --> (60000, 100)

x_train_tf = m.transform(x_train)

# (60000, 100)! reduced the dimensionality!
x_train_tf.shape

# now to go backwards and see the image:
# (60000, 100) * (100, 784) --> (60000, 784)
xback = m.inverse_transform(x_train_tf)
xback.shape

#and back to 28x28
draw_array(xback.reshape(60000,28,28))


# Visualizing the Principal Components (aka the Eigenvectors):
draw_array(m.components_.reshape(100,28,28))


"""
What would happen if you used PCA on a single class?
Reducing the variance of a single class before comparing them to
rest of the reduced dimensions of each other class?
"""

# Since we've reduced the dimensions, now we can use a simpler model:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train_tf, ytrain)

rf.score(x_train_tf, ytrain)

x_test = scaler.transform(xtest.reshape(10000,28*28))
x_test = m.transform(x_test)

# 91% accurate on test data
rf.score(x_test,ytest)


"""
When doing PCA, how many principal components should I use?
"""
#%%
# using all of them we can see where we can safely cut off our dimensions
m784 = PCA(n_components=784)
m784.fit(x_train)

#keeping all the components means this should reduce to all 100% still
m784.explained_variance_ratio_.sum()

#instead plot the cumsum to see how useful each additional element is:
m784.explained_variance_ratio_.cumsum()

import numpy as np
import matplotlib.pyplot as plt
plt.plot(m784.explained_variance_ratio_.cumsum())
