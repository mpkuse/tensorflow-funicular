# A FCN (fully connected network) implementation

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import code
import time
import cv2
import math
import matplotlib.pyplot as plt



input_img = tf.keras.layers.Input( shape=(None,None,3), batch_size=2 )

y = tf.keras.layers.Conv2D( 32, (3,3), padding='same' )(input_img)
y = tf.keras.layers.Conv2D( 32, (3,3), padding='same' )(y)
P1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(y)

x = tf.keras.layers.Conv2D( 64, (3,3), padding='same' )(P1)
x = tf.keras.layers.Conv2D( 64, (3,3), padding='same' )(x)
P2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

sup = tf.keras.layers.Conv2DTranspose( 64, (5,5), strides=2, padding='same' )( P2 )

sup = tf.keras.layers.Add()( [sup, x] )

sup = tf.keras.layers.Conv2DTranspose( 32, (5,5), strides=2, padding='same' )( sup )
sup = tf.keras.layers.Add()( [sup, y] )


model = tf.keras.models.Model( inputs=input_img, outputs=sup )

model.summary()
tf.keras.utils.plot_model( model, show_shapes=True )
