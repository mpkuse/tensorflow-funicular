# Adopted from : https://github.com/Hvass-Labs/TensorFlow-Tutorials

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import code
import time
import cv2
import math
import matplotlib.pyplot as plt

def labels_to_logits( labels ):
    n_classes = len( np.unique( labels ) )
    logits = np.zeros( (labels.shape[0], n_classes) )

    for i in range( len(labels) ):
        logits[i, labels[i] ] = 1.
    return logits

def cmap_imshow( im ):
    assert( len(im.shape) == 2 )
    w_min = im.min()
    w_max = im.max()

    im_gray = ((im - w_min) / (w_max - w_min) * 255).astype('uint8')
    cmap = cv2.applyColorMap(im_gray, cv2.COLORMAP_HOT)
    cmap = cv2.resize(cmap, (0,0), fx=20, fy=20,  interpolation=cv2.INTER_NEAREST)


    return cmap


def plot_conv_weights(weights, input_channel=0):
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int( math.ceil(math.sqrt(num_filters)) )

    # Create figure with a grid of sub-plots.
    # code.interact( local=locals() )
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_conv_output(values):
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int( math.ceil(math.sqrt(num_filters)) )

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


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

#---------------------------------------------------------------------------
# Save Learned Model
if True:
    print 'load pretrained model: model.keras'
    model.load_weights( 'model.keras' )

#---------------------------------------------------------------------------
# Compile and Run iterations
if False:
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
    model.save( 'model.keras' )


#----------------------------------------------------------------------------
# Analyse Layers
print 'len( model.layers ) : ', len( model.layers )
for l in model.layers:
    print l
print '---'
print 'len( model.get_weights() )', len( model.get_weights() )
for w in model.get_weights():
    print w.shape

#-----------------------------------------------------------------------------
# Visualize Filters - W[0] and W[2]
W = model.get_weights()
# W[0] : 5x5x1x16
# W[2] : 5x5x16x36

# print '---'
# cv2.imshow( 'W[0]' , cmap_imshow(W[0][:,:,0,0]) )
# cv2.waitKey(0)

# print W[0].shape #5,5,1,16
# plot_conv_weights( W[0], 0 )
# for i in range( W[2].shape[2] ):
#     plot_conv_weights( W[2], i )


#-----------------------------------------------------------------------------
# Visualize Outputs of Conv
model_conv1 = tf.keras.models.Model( inputs=model.layers[0].input, outputs=model.layers[1].output )
model_conv2 = tf.keras.models.Model( inputs=model.layers[0].input, outputs=model.layers[3].output )

input_image_ary =  x_test[2,:,:]
out_conv1 = model_conv1.predict( input_image_ary.reshape(1,28*28).astype('float32') )
out_conv2 = model_conv2.predict( input_image_ary.reshape(1,28*28).astype('float32') )

cv2.imshow( 'input_image_ary', input_image_ary )
cv2.waitKey(0)

print 'out_conv1.shape', out_conv1.shape
plot_conv_output( out_conv1 )

print 'out_conv2.shape', out_conv2.shape
plot_conv_output( out_conv2 )
code.interact( local=locals() )
quit()



#----------------------------------------------------------------------------
# Testing
pred_outs = model.predict( x_test[0:1000,:,:].reshape(1000,28*28) )
print 'predits: ', pred_outs.argmax( axis=1 )
print 'desired: ', y_test[0:1000]
print 'n_correct = ', ((pred_outs.argmax( axis=1 ) - y_test[0:1000]) == 0 ).sum()
