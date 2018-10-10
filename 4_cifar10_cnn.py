# Loads Cifar10. and fits a simple cnn

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import code
import time
import cv2
import math
import matplotlib.pyplot as plt

from viz_utils import labels_to_logits


#-----------------------------------------------------------------------------
# Data
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train: 50K x 32x32x3
#y_train: 50K x 1

#-----------------------------------------------------------------------------
# Model
model = tf.keras.Sequential()

model.add( tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3) ) )
model.add( tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same' ) )
model.add( tf.keras.layers.MaxPooling2D(pool_size=(2,2)) )

model.add( tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(32,32,1) ) )
model.add( tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same' ) )
model.add( tf.keras.layers.MaxPooling2D(pool_size=(2,2)) )

model.add( tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', input_shape=(32,32,1) ) )
model.add( tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same' ) )
model.add( tf.keras.layers.MaxPooling2D(pool_size=(2,2)) )

model.add( tf.keras.layers.Flatten() )
model.add( tf.keras.layers.Dense(128, activation='relu'))
model.add( tf.keras.layers.Dense(10, activation='softmax'))


model.summary()
tf.keras.utils.plot_model( model, show_shapes=True )


#-----------------------------------------------------------------------------
# Compile
# optimizer = tf.keras.optimizers.Adam(lr=1e-5)
optimizer = tf.keras.optimizers.RMSprop(lr=1e-4)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#----------------------------------------------------------------------------
# Iterations
if True: # Simple 1 shot

    tb = tf.keras.callbacks.TensorBoard( log_dir='cifar10_cnn.logs', histogram_freq=1, write_grads=True, write_images=True )

    history = model.fit(x=x_train.reshape( x_train.shape[0], x_train.shape[1], x_train.shape[2], 3),
              y=labels_to_logits(y_train),
              epochs=5, batch_size=128, verbose=1,
              callbacks=[tb], validation_split=0.1)
    print 'save learned model'
    model.save( 'cifar10_cnn.keras' )
    code.interact( local=locals() )


if False: #
    x_train_1 = x_train[ 0:25000, :, : , : ]
    y_train_1 = y_train[ 0:25000, : ]
    x_train_2 = x_train[ 25000:, :, : , : ]
    y_train_2 = y_train[ 25000:, : ]

    model.fit(x=x_train_1,
              y=labels_to_logits(y_train_1),
              epochs=10, batch_size=128, verbose=2)

    model.fit(x=x_train_2,
              y=labels_to_logits(y_train_2),
              epochs=10, batch_size=128, verbose=2)

    print 'save learned model'
    model.save( 'cifar10_cnn.keras' )


if False:
    print 'load pretrained model'
    model.load_weights( 'cifar10_cnn.keras' )


#---------------------------------------------------------------------------
# Evaluate
score = model.evaluate( x_test, labels_to_logits(y_test), verbose=1 )
print 'Test Loss: ', score[0]
print 'Accuracy : ', score[1]



#---------------------------------------------------------------------------
# Predict
for _ in range(30):
    r = np.random.randint( x_test.shape[0] )
    pred_outs = model.predict( x_test[r:r+1,:,:,:] )
    print 'r=', r
    print 'predicted = ', pred_outs.argmax(),
    print 'ground truth = ', y_test[r],
    print ''
    cv2.imshow( 'test image', x_test[r,:,:,:].astype('uint8') )
    cv2.waitKey(0)
