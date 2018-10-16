"""
    Learning a fully-connect layers with MNIST dataset.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 8th Sept, 2018
"""


import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import code
import time
import cv2


# tf.enable_eager_execution()
# tfe = tf.contrib.eager

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
    model.add( tf.keras.layers.Dense( 392, input_shape=(None,784) ) )
    model.add( tf.keras.layers.Dense( 196 ) )
    model.add( tf.keras.layers.Dense( 98 ) )
    model.add( tf.keras.layers.Dense( 49 ) )
    model.add( tf.keras.layers.Dense( 10 ) )

if False:
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
# Training Op
in_ = tf.placeholder( tf.float32, shape=(None, 28*28) )
logits_out_ = tf.placeholder( tf.float32, shape=(None, 10) )
labels_out_ =  tf.placeholder( tf.int32, shape=(None) )
loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=labels_out_, logits=model(in_) ) )

x_sample = x_train[0:5,:,:].reshape( 5, 28*28 )
y_gt_sample = y_train[0:5]
# y_sample = model( x_sample )


#----------------------------------------------------------------------------
# Session
sess = tf.Session()
tf.keras.backend.set_session( sess )


#----------------------------------------------------------------------------
# Init
init_op = tf.global_variables_initializer()
sess.run(init_op)


#----------------------------------------------------------------------------
# Iterations
# train_step = tf.train.GradientDescentOptimizer(0.00005).minimize(loss) #TODO
train_step = tf.train.AdamOptimizer(0.00005).minimize(loss) #TODO

batch_size = 64
for i in range(100):
    # start_t = time.time()
    bstart = i*batch_size
    bend = (i+1)*batch_size
    print 'using : ', bstart, bend , '--->',

    feed_in = x_train[bstart:bend,:,:].reshape( batch_size, 28*28 )
    feed_lab = y_train[bstart:bend]

    tff_loss, _ = sess.run( [loss,train_step], feed_dict={in_:feed_in  , labels_out_:feed_lab } )
    print i, tff_loss#, ' took %4.2fms' %(time.time() - start_t)



#------------------------------------------------------------------------
# Save Model
saver = tf.train.Saver()
print 'Save Model'
save_path = saver.save(sess, "./mnist.model/model.ckpt")


#-------------------------------------------------------------------------
# Test Model
final_predictions =  sess.run( model(in_), feed_dict={in_: x_test[0:100,:,:].reshape( 100, 28*28 )} )
final_labels = final_predictions.argmax( axis=1 )

for i in range(100):
    cv2.imshow( 'x_test',  x_test[i,:,:] )
    print i, y_test[i], final_labels[i]
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
