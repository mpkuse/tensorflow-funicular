import keras
import numpy as np

model = keras.Sequential()
model.add( keras.layers.Embedding(1000,64, input_length=10) )

input_array = np.random.randint( 1000, size=(32,10) )
model.compile( 'rmsprop', 'mse')
out = model.predict( input_array ) #output.shape = (32,10,64)
