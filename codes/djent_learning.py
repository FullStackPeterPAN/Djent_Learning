from keras.layers.core import Dense, Activation, Dropout


def learning_model(model, input_data, expected_data):
    model.add(Dense(256, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('softmax'))

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_in = input_data
    train_out = expected_data

    print(train_in)
    print(train_out)

    # train the model
    model.fit(train_in, train_out, epochs=100, batch_size=1000)

    # save the model
    model.save("data/model/model.h5")

    # evaluate the model
    loss, accuracy = model.evaluate(train_in, train_out)
    print(loss, accuracy)
