from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)


model = Sequential([
    Dense(4, input_shape=(2,)),
    Activation('sigmoid'),
    Dense(1),
    Activation('sigmoid'),
])

model.compile(optimizer='rmsprop', loss='mse')

model.fit(X, y, epochs=500, batch_size=4)

score = model.evaluate(X, y, batch_size=4)
print(score)

print(model.predict(X))
