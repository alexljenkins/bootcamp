"""
0 : T-shirt/top
1 : Trouser
2 : Pullover
3 : Dress
4 : Coat
5 : Sandal
6 : Shirt
7 : Sneaker
8 : Bag
9 : Ankle boot
"""
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras import backend as K
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import ModelCheckpoint
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

checkpoint_path = "data/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


def array_converter(array):
    """
    Scale down to values between 0 and 1
    """
    array = tf.keras.utils.normalize(array, axis=1)
    array = array[:,:,:,np.newaxis]

    return array

def one_hot_encoder(array):
    """
    Converts the y values into a binary array.
    """
    ohe = np.zeros((array.size, array.max()+1))
    ohe[np.arange(array.size),array] = 1

    return ohe

def one_hot_encoder_v2(x, axis = 0):
    """
    Initial encoder wasn't working,
    despite being the right shape, so
    let's try another method.
    """
    return np.stack(x, axis=axis)

def model_initializer(callback = None):
    """
    Setting the initial state of the neural network.
    """
    # if callback == None:
    # clear any saved models
    K.clear_session()

    # initialize a new model
    model = tf.keras.models.Sequential()

    # add layers
    model.add(tf.keras.layers.Conv2D(24, (3,3), input_shape = (28,28,1), strides=(1, 1),
                                    padding='valid', data_format=None, dilation_rate=(1, 1),
                                    activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros'))
    model.add(tf.keras.layers.Conv2D(24, (3,3), strides=(1, 1),
                                    padding='valid', data_format=None, dilation_rate=(1, 1),
                                    activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(1, 1), padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # compile and return
    model.compile(optimizer='rmsprop',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


    return model


def metrics(model, x_train, y_train, x_test, y_test):
    score = model.evaluate(x_train, y_train, batch_size=1000)
    print(score)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(test_loss, test_acc)

    # pick a random item from the test dataset
    # and display it with the model's predictions
    random_compare = np.random.randint(1,len(y_test))
    print(np.argmax(predictions[random_compare]))
    plt.imshow(x_test[random_compare][:,:,0], cmap=plt.cm.Greys)
    plt.show()


def visualize_network(model):
    """
    Creates a visual representation of the neural network.
    Requires graphviz: http://www.graphviz.org/ and pip install graphviz
    """
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # ann_viz(model, view=True, filename="network.gv", title="MNIST CNN")


if __name__ == '__main__':

    # scale down to values between 0 and 1
    # and add an extra dimension
    x_train = array_converter(x_train)
    x_test = array_converter(x_test)

    # one hot encode our y values
    y_train = one_hot_encoder_v2(y_train)
    y_test = one_hot_encoder_v2(y_test)

    # load the trained model if exists else initialize a new one
    try:
        model = tf.keras.models.load_model('data/saved_model.h5')

    except:
        # initialize
        model = model_initializer()

    # fit
    model.fit(x_train, y_train,
                epochs=5,
                validation_data=(x_test,y_test)) #, batch_size=1000,

    # save model
    model.save('data/saved_model.h5')

    # results
    predictions = model.predict([x_test])
    metrics(model, x_train, y_train, x_test, y_test)
