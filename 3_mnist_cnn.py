# Adopted from Keras Examples
# Also see https://github.com/keras-team/keras/tree/master/examples

import numpy as np
import tensorflow as tf
import code
import time
import cv2
import math
import matplotlib.pyplot as plt
from viz_utils import labels_to_logits
#---------------------------------------------------------------------------
# Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train: 60K x 28 x 28
# y_train: 60K x 1
# x_test : 10K x 28 x 28
# y_test : 10K x 1


#----------------------------------------------------------------------------
# Model
model = tf.keras.Sequential()

model.add( tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1) ) )
model.add( tf.keras.layers.Conv2D(64, (3,3), activation='relu',  padding='same') )
model.add( tf.keras.layers.MaxPooling2D(pool_size=(2,2)) )

model.add( tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', input_shape=(28,28,1) ) )
model.add( tf.keras.layers.Conv2D(128, (3,3), activation='relu',  padding='same') )
model.add( tf.keras.layers.MaxPooling2D(pool_size=(4,4)) )

model.add( tf.keras.layers.Flatten() )

model.add( tf.keras.layers.Dense(128, activation='relu'))
model.add( tf.keras.layers.Dense(10, activation='softmax'))


model.summary()

#-----------------------------------------------------------------------------
# Compile
# optimizer = tf.keras.optimizers.Adam(lr=1e-5)
optimizer = tf.keras.optimizers.RMSprop(lr=1e-4)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#----------------------------------------------------------------------------
# Iterations
if False:

    model.fit(x=x_train.reshape( 60000, 28,28, 1),
              y=labels_to_logits(y_train),
              epochs=10, batch_size=128, verbose=2)


    print 'save learned model'
    model.save( 'mnist_cnn.keras' )


#---------------------------------------------------------------------------
# train_on_batch
if True:
    for i in range(1000):
        r = np.random.randint( x_train.shape[0], size=32 )
        # code.interact( local=locals() )
        tr_loss = model.train_on_batch( x=x_train[r,:,:].reshape(len(r),28,28,1), y=labels_to_logits(y_train[r], n_classes=10) )
        print i, tr_loss

tf.keras.utils.plot_model( model, show_shapes=True )
quit()
#---------------------------------------------------------------------------
# Load pretrained model
if False:
    print 'load pretrained model'
    model.load_weights( 'mnist_cnn.keras' )


#---------------------------------------------------------------------------
# Evaluate
score = model.evaluate( x_test.reshape( 10000, 28,28,1 ), labels_to_logits(y_test), verbose=1 )
print 'Test Loss: ', score[0]
print 'Accuracy : ', score[1]


#---------------------------------------------------------------------------
# Predict
for _ in range(30):
    r = np.random.randint( x_test.shape[0] )
    pred_outs = model.predict( x_test[r,:,:].reshape( 1, 28,28,1) )
    print 'r=', r
    print 'predicted = ', pred_outs.argmax(),
    print 'ground truth = ', y_test[r],
    print ''
    cv2.imshow( 'test image', x_test[r,:,:].astype('uint8') )
    cv2.waitKey(0)
