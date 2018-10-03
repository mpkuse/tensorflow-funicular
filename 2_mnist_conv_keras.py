# Adopted from : https://github.com/Hvass-Labs/TensorFlow-Tutorials

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import code
import time
import cv2

def labels_to_logits( labels ):
    n_classes = len( np.unique( labels ) )
    logits = np.zeros( (labels.shape[0], n_classes) )

    for i in range( len(labels) ):
        logits[i, labels[i] ] = 1.
    return logits

#---------------------------------------------------------------------------
# Data
# Turns out Keras has functions to import the toy datasets that are in use.
# https://keras.io/datasets/
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#---------------------------------------------------------------------------
# Build Model - Fully Connected
model = tf.keras.Sequential()

if True:
    model.add(tf.keras.layers.InputLayer(input_shape=(28*28,) ) )
    model.add(tf.keras.layers.Reshape((28,28,1)))


    model.add( tf.keras.layers.Conv2D(kernel_size=(5,5), strides=1, filters=16, padding='same',
                     activation='relu', name='layer_conv1'), )
    model.add( tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add( tf.keras.layers.Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2'))
    model.add( tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    model.add( tf.keras.layers.Flatten())
    model.add( tf.keras.layers.Dense(128, activation='relu'))
    model.add( tf.keras.layers.Dense(10, activation='softmax'))



model.summary()

if True:
    #---------------------------------------------------------------------------
    # Save Learned Model
    print 'load pretrained model: model.keras'
    model.load_weights( 'model.keras' )

if False:
    #---------------------------------------------------------------------------
    # Compile
    optimizer = tf.keras.optimizers.Adam(lr=1e-5)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #---------------------------------------------------------------------------
    # Iterations
    model.fit(x=x_train.reshape( 60000, 28*28),
              y=labels_to_logits(y_train),
              epochs=10, batch_size=128, verbose=2)

    #---------------------------------------------------------------------------
    # Save Learned Model
    print 'Save Model (Keras): model.keras'
    model.save( 'model.keras')


#----------------------------------------------------------------------------
# Analyse Layers
code.interact( local=locals() , banner='weights')
print 'len( model.layers ) : ', len( model.layers )
for l in model.layers:
    print l
print '---'
print 'len( model.get_weights() )', len( model.get_weights() )
for w in model.get_weights():
    print w.shape 

#----------------------------------------------------------------------------
# Testing
pred_outs = model.predict( x_test[0:1000,:,:].reshape(1000,28*28) )
print 'predits: ', pred_outs.argmax( axis=1 )
print 'desired: ', y_test[0:1000]
print 'n_correct = ', ((pred_outs.argmax( axis=1 ) - y_test[0:1000]) == 0 ).sum()
