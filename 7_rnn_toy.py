""" Learning RNNs. Based on Karpaty's blog post
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 16th Oct, 1990
"""


# import keras
import tensorflow as tf
import numpy as np
import code

#---
#--- Data
#---
learning_data = 'hello'
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]
nop = [1,1,1,1]

# 4 data points for 'hello'
x = []
y = []
# h ==> e
x.append( [ h, nop, nop, nop ] )
y.append( e )

# he ==> l
x.append( [h, e, nop, nop])
y.append( l )

# hel ==> l
x.append( [h, e, l, nop])
y.append( l )

# hell ==> o
x.append( [h, e, l, l])
y.append( o )

x = np.array( x )
y = np.array( y )
print( 'x.shape', x.shape )
print( 'y.shape', y.shape )
# code.interact( local=locals() )

#---
#--- Construct Network
#---
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(7, input_shape=(4, 4)))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

model.summary()
tf.keras.utils.plot_model( model, show_shapes=True )


optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(x=x, y=y, batch_size=4, epochs=20 )
