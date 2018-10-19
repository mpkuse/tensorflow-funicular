# http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/
# http://adventuresinmachinelearning.com/keras-lstm-tutorial/

import keras
import io
import numpy as np
import code


class KerasBatchGenerator( keras.utils.Sequence ):
    def __init__(self):
        #---
        #--- Make Dataset
        #---
        path ='ptb_txt_data/ptb.train.txt'
        # path ='ptb_txt_data/ptb.test.txt'
        # path ='ptb_txt_data/ptb.valid.txt'
        with io.open(path, encoding='utf-8') as f:
            print 'open file', path
            text = f.read().lower().replace( '\n', '<eos>')
        print('corpus length:', len(text))

        # corpus as list of words
        words = text.split()
        print( 'Total words: ', len(words) )

        # Vocabulary
        vocab = sorted( list( set( words)  ) )
        print( 'vocab size: ', len(vocab) )

        word_indx = dict((c, i) for i, c in enumerate(vocab))
        indx_word = dict((i, c) for i, c in enumerate(vocab))

        # corpus as a list of numbers
        words_nums = np.array( [ word_indx[v] for v in words ] )

        self.vocab = vocab
        self.word_indx = word_indx
        self.indx_word = indx_word
        self.words_nums = words_nums

        self.vocab_size = len(vocab)
        self.curr_pos = 0
        self.skip_step = 5
        self.t_steps = 30
        self.batch_size = 32


    def __len__(self):
        return len(self.words_nums) // (self.skip_step*self.batch_size)

    # def generate( self ):
    def __getitem__( self, idx ):
        e_start = idx*self.skip_step*self.batch_size

        # print '---idx=', idx

        X = []
        Y = []
        for i in range( self.batch_size ):
            s = e_start+self.skip_step*i
            e = e_start+self.skip_step*i + self.t_steps

            # if i == 0 or i==self.batch_size-1:
            #     print i, 'start=', s,
            #     print '\tend=', e
            # if i==1:
            #     print '\t...'

            a = self.words_nums[ s:e ]
            b = self.words_nums[ s+1:e+1 ]
            X.append( a )
            Y.append( keras.utils.to_categorical( b, self.vocab_size ) )
        X = np.array( X ) #32 x 30 ints
        Y = np.array( Y ) #32 x 30 x 10000
        return X, Y



if __name__ == '__main__':
    gen = KerasBatchGenerator()
    # X, Y = gen.generate()

    t_steps = gen.t_steps
    vocab_size = gen.vocab_size

    # do gen[0], gen[1] ... to get the items.
    # code.interact( local=locals() )
    # quit()

    #---
    #--- Make Model
    #---
    model = keras.Sequential()
    model.add( keras.layers.Embedding( vocab_size, output_dim=64, input_length=t_steps ))
    model.add(keras.layers.LSTM(512, return_sequences=True ) )
    model.add(keras.layers.LSTM(512, return_sequences=True ) )
    model.add( keras.layers.TimeDistributed( keras.layers.Dense( vocab_size, activation='softmax' ) ) )
    # model.add( keras.layers.Activation('softmax') )


    model.summary()
    keras.utils.plot_model( model, show_shapes=True )


    model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'] )
    # model.fit(x=X, y=Y, batch_size=4, epochs=10, validation_split=0.1, initial_epoch=0 )
    model.fit_generator( gen, epochs=40, verbose=1 )
    model.save( 'model.keras')
