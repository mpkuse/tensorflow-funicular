""" Sample usage of LSTMCell

        https://github.com/farizrahman4u/recurrentshop

        toy example from : http://karpathy.github.io/2015/05/21/rnn-effectiveness/

            Author  : Manohar Kuse <mpkuse@connect.ust.hk>
            Created : 22nd Oct, 2018
"""
import keras
import numpy as np
import code

#---
#--- Data
#---
learning_data = 'hello'
h = [1,0,0,0,0]
e = [0,1,0,0,0]
l = [0,0,1,0,0]
o = [0,0,0,1,0]
nop = [0,0,0,0,1]

# 4 data points for 'hello'
Dx = []
Dy = []
# h ==> e
Dx.append( [ h, nop, nop, nop ] )
Dy.append( e )

# he ==> l
Dx.append( [h, e, nop, nop])
Dy.append( l )

# hel ==> l
Dx.append( [h, e, l, nop])
Dy.append( l )

# hell ==> o
Dx.append( [h, e, l, l])
Dy.append( o )

Dx = np.array( Dx )
Dy = np.array( Dy )
print 'Dx.shape', Dx.shape
print 'Dy.shape', Dy.shape
# code.interact( local=locals() )


if True:
    #---
    #--- Construct Network (from scratch)
    #---
    from keras.layers import Input, Dense, Activation, add

    from recurrentshop import *
    x_t = Input(shape=(5,)) # The input to the RNN at time t
    h_tm1 = Input(shape=(20,))  # Previous hidden state

    # Compute new hidden state
    h_t = add([Dense(20)(x_t), Dense(20, use_bias=False)(h_tm1)])

    # tanh activation
    h_t = Activation('tanh')(h_t)

    y_t = Dense(5, activation='softmax')( h_t )
    # Build the RNN
    # RecurrentModel is a standard Keras `Recurrent` layer.
    # RecurrentModel also accepts arguments such as unroll, return_sequences etc
    rnn = RecurrentModel(input=x_t, initial_states=[h_tm1], output=y_t, final_states=[h_t])

    # return_sequences is False by default
    # so it only returns the last h_t state

    # Build a Keras Model using our RNN layer
    # input dimensions are (Time_steps, Depth)
    x = Input(shape=(4,5))
    y = rnn(x)
    model = keras.models.Model(x, y)


    # Run the RNN over a random sequence
    # Don't forget the batch shape when calling the model!
    out = model.predict(np.random.random((1, 4, 5)))

    print(out.shape)#->(1,10)


    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x=Dx, y=Dy, batch_size=4, epochs=20 )

    quit()

if False:
    from keras.layers import Input, add, Activation
    from recurrentshop import LSTMCell, GRUCell, RecurrentModel
    # RNNCell - Using RNNCells in Functional API
    input = Input( shape=(5,) )
    state1_tm1 = Input((10,))
    state2_tm1 = Input((10,))
    state3_tm1 = Input((10,))

    lstm_output, state1_t, state2_t = LSTMCell(10)([input, state1_tm1, state2_tm1])
    gru_output, state3_t = GRUCell(10)([input, state3_tm1])


    output = add([lstm_output, gru_output])
    output = Activation('tanh')(output)

    rnn = RecurrentModel(input=input, initial_states=[state1_tm1, state2_tm1, state3_tm1], output=output, final_states=[state1_t, state2_t, state3_t])
