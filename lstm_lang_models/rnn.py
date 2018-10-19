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
        return len(self.words_nums) // (self.t_steps*self.batch_size*self.skip_step)

    # def generate( self ):
    def __getitem__( self, idx ):
        while True:
            X = []
            Y = []
            for i in range( self.batch_size ): #generate 32 training samples each time
                if self.curr_pos + self.t_steps >= len( self.words_nums ):
                    self.curr_pos = 0

                a = self.words_nums[ self.curr_pos : self.curr_pos+self.t_steps ]
                # b = self.words_nums[ self.curr_pos+self.t_steps ]
                b_ak = self.words_nums[ self.curr_pos+1 : self.curr_pos+self.t_steps+1 ]

                X.append( a )
                # b_1hot = np.zeros( self.vocab_size )
                # b_1hot[b] = 1.
                # Y.append( b_1hot )
                Y.append( keras.utils.to_categorical( b_ak, self.vocab_size ) )


                self.curr_pos += self.t_steps
            X = np.array( X ) #32 x 30 ints
            Y = np.array( Y ) #32 x 30 x 10000
            # code.interact( local=locals() )
            # yield X, Y
            return X, Y




if __name__ == '__main__':
    gen = KerasBatchGenerator()
    # X, Y = gen.generate()

    t_steps = gen.t_steps
    vocab_size = gen.vocab_size

    # do gen[0], gen[1] ... to get the items.
    # code.interact( local=locals() )

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
