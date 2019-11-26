import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=50, noise=0.2, random_state=42)

#Visualizing the data
plt.scatter(X[:,0],y)
plt.scatter(X[:,1],y)
plt.show()


plt.scatter(X[:,0],X[:,1], c = y)
plt.show()



#%%

#Add a bias column, fill with 1
# X = np.hstack([X, np.ones((X.shape[0], 1))])

#defining the sigmoid function
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


# %% defining the feed forward function
def feed_forward(X, weights1, weights2):
    """

    """
    # add a bias column of ones
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])

    # calculate the dot product of X and the weights of the first layer
    input_layer = np.dot(Xb, weights1)
    # apply the sigmoid function on the result
    hidden_output = sigmoid(input_layer)
    # print(hidden_output.shape)
    # append bias of ones
    hidden_output_b = np.hstack([hidden_output, np.ones((hidden_output.shape[0], 1))])

    # calculate the dot product of the first layer's output and the weights of the second layer
    layer2 = np.dot(hidden_output_b, weights2)

    # print(hidden_output.shape)
    # apply sigmoid and bias
    output = sigmoid(layer2)

    return Xb, hidden_output_b, output


# %% testing the feed_forward function

# weights1 = np.ones((3,2))
# weights2 = np.ones((3,1))
#
# X_out, out1, out2 = feed_forward(X, weights1, weights2)
#
# assert out1.shape == (50, 1)
# assert out2.shape == (50, 1)
# assert X_out.shape == (50,3)

# %% Backpropagation steps

def error_calculator(ypred, ytrue):
    """
    Calculates the error of each element with the formula:
    Error = (yhat - ytrue) * loss
    where loss > 0 defined below.
    """
    loss = -(ytrue * np.log(ypred) + (1-ytrue)) * np.log(1-ypred)
    error = (ypred - ytrue) * loss

    return error

def y_gradient_calculator(ypred, error):
    """
    How do we have to modify the weights to minimize our error?:
    activation_funtion'(hidden_output . output_weights)*error
    Should we adjust a weight, and in which direction?
    """
    gradient_of_y_vector = sigmoid(ypred) * (1-sigmoid(ypred)) * error

    return gradient_of_y_vector


def output_weights_modifier(gradient_of_y_vector, hidden_output, learning_rate):
    """
    Finding the amount we should adjust each weight
    """
    change_to_output_weights = -(np.dot(gradient_of_y_vector, hidden_output)) * learning_rate
    gradient_of_y_vector.shape
    hidden_output.shape
    change_to_output_weights.shape
    output_weights.shape

    return change_to_output_weights


def h_gradient_calculator(gradient_of_y_vector, hidden_weights, output_weights):
    """
    Finds the gradient of the hidden weights with:
    activation_funtion'(X . hidden_weights)*(gradient_of_y_vector . output_weights)
    """
    gradient_of_y_vector = gradient_of_y_vector.reshape(50,1)
    gradient_of_h = sigmoid(np.dot(X,hidden_weights) * (1-sigmoid(np.dot(X,hidden_weights)))) * np.dot(gradient_of_y_vector, output_weights[:2].T) #output_weights[:2].T

    return gradient_of_h


def hidden_weights_modifier(X, gradient_of_h, learning_rate):
    """
    changes_to_hidden_weights = -(gradent_of_h . X) * learning_rate
    """
    changes_to_hidden_weights = -np.dot(X.T, gradient_of_h) * learning_rate

    return changes_to_hidden_weights




if __name__ == '__main__':
    X, y = make_moons(n_samples=50, noise=0.2, random_state=42)
    hidden_weights = np.ones((3,2))
    output_weights = np.ones((3,1))
    learning_rate = 0.1

    X, hidden_output, ypred = feed_forward(X, hidden_weights, output_weights)

    # checking shapes
    assert X.shape == (50,3)

    # y as (50,1) seems to explode our error function into (50,50)
    ypred = ypred.flatten()
    assert ypred.shape == (50,)
    assert hidden_output.shape == (50,3)

    error = error_calculator(ypred, y)
    assert error.shape == (50,)

    gradient_of_y_vector = y_gradient_calculator(ypred, error)
    assert gradient_of_y_vector.shape == (50,)

    change_to_output_weights = output_weights_modifier(gradient_of_y_vector, hidden_output, learning_rate)
    change_to_output_weights = change_to_output_weights.reshape(3,1)
    assert change_to_output_weights.shape == (3,1)
    output_weights += change_to_output_weights

    gradient_of_h = h_gradient_calculator(gradient_of_y_vector, hidden_weights, output_weights)
    assert gradient_of_h.shape == (50,2)

    changes_to_hidden_weights = hidden_weights_modifier(X, gradient_of_h, learning_rate)
    assert changes_to_hidden_weights.shape == (3,2)
    hidden_weights += changes_to_hidden_weights

    # round the answers to a predition if we want to change the threshold.
    # yhat = [1 if x > 0.5 else 0 for x in ypred]
    # else use np.round since this is much faster
