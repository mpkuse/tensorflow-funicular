# Loading pretrained nets

import keras
import numpy as np


if True : # Load resnet50
    model = keras.applications.resnet50.ResNet50( weights='imagenet' )

    model.summary()
    keras.utils.plot_model( model, show_shapes=True )

    img = keras.preprocessing.image.load_img( 'elephant.jpeg', target_size=(224,224) )
    x = keras.preprocessing.image.img_to_array( img )
    x = np.expand_dims( x, axis=0 )

    preds = model.predict( x )
    print keras.applications.resnet50.decode_predictions( preds )


model = keras.applications.vgg16.VGG16( weights='imagenet', include_top=False )
model.summary()
keras.utils.plot_model( model, show_shapes=True )

img = keras.preprocessing.image.load_img( 'elephant.jpeg', target_size=(224,224) )
x = keras.preprocessing.image.img_to_array( img )
x = np.expand_dims( x, axis=0 )
