# Trying out the keras functional API

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import code
import time
import cv2
import math
import matplotlib.pyplot as plt

from viz_utils import labels_to_logits


if False: #trying out functional model API of keras
    a = tf.keras.layers.Input( shape=(32,) )
    b = tf.keras.layers.Dense(32)(a)
    c = tf.keras.layers.Dense(64)(a)
    d = tf.keras.layers.Dense(32)(c)

    # e = tf.keras.layers.Add()( [d,b] )
    e = tf.keras.layers.Dot(axes=1)( [d,b] )

    model = tf.keras.models.Model( inputs=a, outputs=e )

if False: # Reuse of a layer
    a = tf.keras.layers.Input( shape=(32,) )
    b = tf.keras.layers.Input( shape=(32,) )

    c = tf.keras.layers.Dense(32)
    d = c(a)
    e = c(b)
    model = tf.keras.models.Model( inputs=[a,b], outputs=[d,e])


# Inception
if False:
    input_img = tf.keras.layers.Input( shape=(256,256,3) )

    tower_1 = tf.keras.layers.Conv2D( 64, (1,1), padding='same', activation='relu' )( input_img )
    tower_1 = tf.keras.layers.Conv2D( 64, (3,3), padding='same', activation='relu' )( tower_1 )

    tower_2 = tf.keras.layers.Conv2D( 64, (1,1), padding='same', activation='relu' )( input_img )
    tower_2 = tf.keras.layers.Conv2D( 64, (5,5), padding='same', activation='relu' )( tower_2 )

    tower_3 = tf.keras.layers.MaxPooling2D( (3,3), strides=(1,1), padding='same' )( input_img )
    tower_3 = tf.keras.layers.Conv2D( 64, (1,1), padding='same', activation='relu' )( tower_3 )

    output = tf.keras.layers.concatenate( [tower_1, tower_2, tower_3], axis=1 )

    model = tf.keras.models.Model( inputs=input_img, outputs=output)


# Residual Connection - basics
if False:
    input_img = tf.keras.layers.Input( shape=(256,256,3) )

    y = tf.keras.layers.Conv2D( 3, (3,3), padding='same' )(input_img)
    z = tf.keras.layers.add( [input_img,y] )
    model = tf.keras.models.Model( inputs=input_img, outputs=z)


# Resnet V1
if False:
    input_img = tf.keras.layers.Input( shape=(512,512,256) )

    a = tf.keras.layers.Conv2D( 64, (1,1), padding='same', activation='relu' )( input_img )
    a = tf.keras.layers.Conv2D( 64, (3,3), padding='same', activation='relu' )( a )
    a = tf.keras.layers.Conv2D( 256, (1,1), padding='same', activation='relu' )( a )

    z = tf.keras.layers.Add()( [input_img,a] )
    model = tf.keras.models.Model( inputs=input_img, outputs=z)


# ResNeXt
if False:
    input_ = tf.keras.layers.Input( shape=(512,512,256) )

    L = []
    for i in range(5):
        a = tf.keras.layers.Conv2D( 4, (1,1), padding='same', activation='relu' )( input_ )
        a = tf.keras.layers.Conv2D( 4, (3,3), padding='same', activation='relu' )( a )
        a = tf.keras.layers.Conv2D( 256, (1,1), padding='same', activation='relu' )( a )
        L.append( a )
    c = tf.keras.layers.Add()( L )

    z = tf.keras.layers.Add()( [input_,c] )
    model = tf.keras.models.Model( inputs=input_, outputs=z)


model.summary()
tf.keras.utils.plot_model( model, show_shapes=False )
