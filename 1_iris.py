"""
    Learning a fully-connect layer with IRIS dataset.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 8th Sept, 2018
"""

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import code

# Eager Execution was introduced in tf1.08. It is useful but I am not using it.
# tf.enable_eager_execution()
# tfe = tf.contrib.eager

def labels_to_logits( labels ):
    n_classes = len( np.unique( labels ) )
    logits = np.zeros( (labels.shape[0], n_classes) )

    for i in range( len(labels) ):
        logits[i, labels[i] ] = 1.
    return logits

def load_iris_data():
    pass
    try:
        X = np.loadtxt( 'iris.data/iris_training.csv', skiprows=1, delimiter=',' )
    except:
        print 'iris_training.csv file not found.\nDownload from : https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/monitors/iris_training.csv'
        print """
            mkdir iris.data
            wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/monitors/iris_training.csv -P iris.data/
        """

        quit()
    labels = X[:,-1]
    feat = X[:,0:-1]

    # make logits from labels
    logits = labels_to_logits( labels.astype('int') )

    return feat, labels, logits






if __name__=='__main__':
    feat, desired_labels, desired_logits = load_iris_data()



    #--------------------------------------------------------------------
    # Build Neural Net - Keras.

    # Way-1
    # model = tf.keras.Sequential( [ tf.keras.layers.Dense( 10, input_shape=(None, 4) ),\
    #                  tf.keras.layers.Dense( 10 ),\
    #                  tf.keras.layers.Dense( 3 ),\
    # ] )

    # Way-2
    model = tf.keras.Sequential()
    model.add( tf.keras.layers.Dense( 10, input_shape=(None, 4) ) )
    model.add( tf.keras.layers.Dense( 10 ) )
    model.add( tf.keras.layers.Dense( 3 )  )


    model.summary()


    #------------------------------------------------------------------------
    # Training Op
    in_feat = tf.placeholder( tf.float32, shape=(None,4) )
    in_logits = tf.placeholder( tf.float32, shape=(None,3) )
    in_labels = tf.placeholder( tf.int32, shape=(None) )
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=model(in_feat), logits=in_logits)
    # loss = tf.reduce_mean(tf.keras.objectives.categorical_crossentropy(in_logits, model(in_feat)))
    # loss = tf.reduce_mean( model(in_feat) - in_logits )
    loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=in_labels, logits=model(in_feat) ) )


    #-------------------------------------------------------------------
    # Session
    sess = tf.Session()
    tf.keras.backend.set_session( sess )


    #----------------------------------------------------------------------
    # Initialize Optimization Variables - Xavier
    if True:
        print 'Random Init'
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    #-----------------------------------------------------------------------
    # Restore Pretrained Model
    if False:
        print 'Restore Model'
        saver = tf.train.Saver()
        saver.restore(sess, "./iris.model/model.ckpt")


    #-----------------------------------------------------------------------
    # Iterations
    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
    for i in range(100):
        tff_loss, _ =  sess.run( [loss, train_step], feed_dict={in_feat: feat , in_logits:desired_logits, in_labels: desired_labels } )
        print 'Iteration#%di: Loss=%4.4f' %(i, tff_loss)



    #------------------------------------------------------------------------
    # Save Model
    saver = tf.train.Saver()
    print 'Save Model'
    save_path = saver.save(sess, "./iris.model/model.ckpt")


    #------------------------------------------------------------------------
    # Run Trained Model - Test Phase
    final_predictions =  sess.run( model(in_feat), feed_dict={in_feat: feat} )
    print 'final_predictions: ', final_predictions.argmax(axis=1)
    print 'desired_labels: ', desired_labels
