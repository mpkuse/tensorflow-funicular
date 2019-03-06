# Adopted from "Getting Started : 30 seconds to Keras' from keras.io

from keras.models import Sequential
model = Sequential()

from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.summary()

import numpy as np
x_train = np.random.rand(1024,100) #1024 instances each with 100d data 
y_train = np.random.rand(1024,10)
model.fit(x_train, y_train, epochs=15, batch_size=32)

