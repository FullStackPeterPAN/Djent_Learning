import wave
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import scipy.io as sio
import numpy as np
import os

x = np.random.randint(10, size=[1, 12])
y = np.random.randint(10, size=[1, 12])
model = Sequential()
model.add(Dense(2, input_dim=12, activation='relu'))
model.add(Dense(12, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])  # compile the model

# train the model
model.fit(x, y, epochs=1, batch_size=1)


# evaluate the model
loss, accuracy = model.evaluate(x, y, batch_size=1)

print(x)
print(y)
print(loss, accuracy)
