""" Taken a corpus of text as input and predicts the next word """

import keras
import numpy as np
import code
import io


#---
#--- Get Remote File
#---
path = keras.utils.data_utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()#.replace( '\n', ' ')
print('corpus length:', len(text))


#---
#--- Make Dataset
#---
chars = sorted( list( set( text ) ) )

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Window of size W on the text
W = 40
x = []
y = []
for i in range( 0, len(text)-W, 10 ): # make this 3 on pc with larger memory
    if i%10000 == 0:
        print 'process char#', i , ' of ', len(text)
    # print i, i+W
    cur_text = text[i:i+W]
    next_char = text[i+W]
    next_char_1hot = np.zeros( len(chars) )
    next_char_1hot[ char_indices[next_char]  ] = 1.


    cur_text_1hot = []
    for ch in cur_text:
        ch_1hot = np.zeros( len(chars) )
        ch_1hot[ char_indices[ch] ] = 1.

        cur_text_1hot.append( ch_1hot )
    cur_text_1hot = np.array( cur_text_1hot )

    x.append( cur_text_1hot )
    y.append( next_char_1hot )
    # code.interact( local=locals() )
    # quit()
x = np.array( x )
y = np.array( y )
print 'x.shape', x.shape
print 'y.shape', y.shape

#---
#--- Construct Network
#---
model = keras.Sequential()
model.add(keras.layers.LSTM(128, input_shape=( x.shape[1], x.shape[2])))
model.add(keras.layers.Dense( x.shape[2], activation='softmax'))
model.load_weights( 'model_text.keras')

model.summary()
keras.utils.plot_model( model, show_shapes=True )

#---
#--- Fit
#---
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
model.fit(x=x, y=y, batch_size=256, epochs=120, validation_split=0.1, initial_epoch=60 )
model.save('model_text.keras')
