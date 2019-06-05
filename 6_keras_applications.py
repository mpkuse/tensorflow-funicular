# Loading pretrained nets

# import keras
import tensorflow as tf
import numpy as np


if True : # Load resnet50
    model = tf.keras.applications.resnet50.ResNet50( weights='imagenet' )

    model.summary()
    tf.keras.utils.plot_model( model, show_shapes=True )

    img = tf.keras.preprocessing.image.load_img( 'elephant.jpeg', target_size=(224,224) )
    x = tf.keras.preprocessing.image.img_to_array( img )
    x = np.expand_dims( x, axis=0 )

    preds = model.predict( x )
    print( tf.keras.applications.resnet50.decode_predictions( preds ) )


model = tf.keras.applications.vgg16.VGG16( weights='imagenet', include_top=False )
model.summary()
tf.keras.utils.plot_model( model, show_shapes=True )

img = tf.keras.preprocessing.image.load_img( 'elephant.jpeg', target_size=(224,224) )
x = tf.keras.preprocessing.image.img_to_array( img )
x = np.expand_dims( x, axis=0 )
