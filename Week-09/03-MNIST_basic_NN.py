import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

from keras.datasets import mnist

from keras import backend as K
# K.clear_session()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%% visualizing the data
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.imshow(x_train[i], cmap=plt.cm.Greys)
#     plt.axis('off')

#%%


def df_normalizer(df):
    """
    Scale down to values between 0 and 1
    """
    df = tf.keras.utils.normalize(df, axis=1)

    return df


def model_initializer():
    """
    Setting the initial state of the neural network.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(128, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.elu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


    model.compile(optimizer='rmsprop',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model


def metrics(model, x_test, y_test):

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)

    random_compare = np.randint(1,len(y_test))

    print(np.argmax(predictions[random_compare]))
    plt.imshow(x_test[random_compare], cmap=plt.cm.Greys)
    plt.show()


if __name__ == '__main__':

    # scale down to values between 0 and 1
    x_train = df_normalizer(x_train)
    x_test = df_normalizer(x_test)

    model = model_initializer()

    model.fit(x_train, y_train, epochs=15) #, batch_size=1000
    predictions = model.predict([x_test])

    metrics(model, x_test, y_test)
