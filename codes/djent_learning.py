# import necessary stuff
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import scipy.io as sio
import numpy as np

# a sequential model for testing
model = Sequential()


def learning_model(input_data, expected_data, dim):
    model.add(Dense(1, input_dim=dim, activation='relu'))
    model.add(Dense(dim, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   # compile the model

    train_in = input_data
    train_out = expected_data.T

    # train the model
    model.fit(train_in, train_out, epochs=1, batch_size=1)

    # save the model
    model.save("F:/WorkSpace/Djent_Learning/codes/data/model/model.h5")

    # evaluate the model
    loss, accuracy = model.evaluate(train_in, train_out)
    print(loss, accuracy)
