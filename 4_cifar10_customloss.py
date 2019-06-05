# Loads Cifar10. and fits a simple cnn. Uses a custom_loss
# Custom loss has been DIY softmax loss (cross entropy) so can verify with original

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
# import keras
# from keras import backend as K
import code
import time
import cv2
import math
# import matplotlib.pyplot as plt

# from viz_utils import labels_to_logits
def labels_to_logits( labels, n_classes=None ):
    if n_classes is None:
        n_classes = len( np.unique( labels ) )

    logits = np.zeros( (labels.shape[0], n_classes) )

    for i in range( len(labels) ):
        logits[i, labels[i] ] = 1.
    return logits

def custom_loss( y_true, y_pred ):
# def custom_loss( params ):
    # y_true, y_pred = params
    # pass
    # code.interact( local=locals() )
    u = -tf.keras.backend.sum( y_true * tf.keras.backend.log(y_pred+1E-6), -1 ) # This is the defination of cross-entropy. basically softmax's log multiply by target
    return tf.keras.backend.maximum( 0., u )


if False:  # Play with custom_loss
    y_true = tf.keras.layers.Input( shape=(10,) )
    y_pred = tf.keras.layers.Input( shape=(10,) )

    # u = custom_loss( y_true, y_pred )
    u = tf.keras.layers.Lambda(custom_loss)( [y_true, y_pred] )
    model = tf.keras.models.Model( inputs=[y_true,y_pred], outputs=u )


    model.summary()
    tf.keras.utils.plot_model( model, show_shapes=True )

    a = np.zeros( (2,10) )
    a[0,1] = 1
    a[1,9] = 1
    b = np.zeros( (2,10) ) #np.random.randint( 10, size=(2,10) )
    b[0,4] = 0.05; b[0,1] = 0.95
    b[1,8] = 1
    out = model.predict( [ a,b ])
    quit()


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
              # loss='categorical_crossentropy',
              loss=custom_loss,
              metrics=['accuracy'])

model.fit( x=x_train, y=labels_to_logits( y_train ),
            epochs=5, batch_size=32, verbose=1, validation_split=0.1 )

model.save( 'cifar10_cnn_customloss.keras' )
#
# #----------------------------------------------------------------------------
# # Iterations
# if True: # Simple 1 shot
#
#     tb = tf.keras.callbacks.TensorBoard( log_dir='cifar10_cnn.logs', histogram_freq=1, write_grads=True, write_images=True )
#
#     history = model.fit(x=x_train.reshape( x_train.shape[0], x_train.shape[1], x_train.shape[2], 3),
#               y=labels_to_logits(y_train),
#               epochs=5, batch_size=128, verbose=1,
#               callbacks=[tb], validation_split=0.1)
#     print 'save learned model'
#     model.save( 'cifar10_cnn.keras' )
#     code.interact( local=locals() )
#
#
# if False: #
#     x_train_1 = x_train[ 0:25000, :, : , : ]
#     y_train_1 = y_train[ 0:25000, : ]
#     x_train_2 = x_train[ 25000:, :, : , : ]
#     y_train_2 = y_train[ 25000:, : ]
#
#     model.fit(x=x_train_1,
#               y=labels_to_logits(y_train_1),
#               epochs=10, batch_size=128, verbose=2)
#
#     model.fit(x=x_train_2,
#               y=labels_to_logits(y_train_2),
#               epochs=10, batch_size=128, verbose=2)
#
#     print 'save learned model'
#     model.save( 'cifar10_cnn.keras' )
#
#
# if False:
#     print 'load pretrained model'
#     model.load_weights( 'cifar10_cnn.keras' )
#
#
# #---------------------------------------------------------------------------
# # Evaluate
# score = model.evaluate( x_test, labels_to_logits(y_test), verbose=1 )
# print 'Test Loss: ', score[0]
# print 'Accuracy : ', score[1]
#
#
#
# #---------------------------------------------------------------------------
# # Predict
# for _ in range(30):
#     r = np.random.randint( x_test.shape[0] )
#     pred_outs = model.predict( x_test[r:r+1,:,:,:] )
#     print 'r=', r
#     print 'predicted = ', pred_outs.argmax(),
#     print 'ground truth = ', y_test[r],
#     print ''
#     cv2.imshow( 'test image', x_test[r,:,:,:].astype('uint8') )
#     cv2.waitKey(0)
