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
for i in range( 0, len(text)-W, 5 ): # make this 3 on pc with larger memory
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
model.add(keras.layers.LSTM(512, input_shape=( x.shape[1], x.shape[2]), return_sequences=True ))
model.add(keras.layers.LSTM(512 ) )
model.add(keras.layers.Dense( 220, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001) ) )
model.add(keras.layers.Dense( x.shape[2], activation='softmax', kernel_regularizer=keras.regularizers.l2(0.0001) ) )


model.summary()
keras.utils.plot_model( model, show_shapes=True )

#---
#--- Fit
#---
# optimizer = keras.optimizers.RMSprop(lr=0.0005)
optimizer = keras.optimizers.Adadelta(  )

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )
if False:
    # model.load_weights( 'model_text.keras')
    model.fit(x=x, y=y, batch_size=512, epochs=40, validation_split=0.1, initial_epoch=0 )
    model.save('model_text.keras')
else:
    model.load_weights( 'model_text.keras' )


#---
#--- Generate Text
#---
ALL_GEN = []
if True: # Use one of the training data
    seed = np.expand_dims( x[10], 0 ) #need a seed to start generation.
if True:
    # random
    seed = np.zeros( (1,W, len(chars)) )
    for i in range(W):
        r = np.random.randint( 27, 52 )
        seed[0,i,r] = 1
    orf_seed = seed

print 'seed.shape=', seed.shape
assert( len(seed.shape) == 3 and seed.shape[0] == 1 and seed.shape[1] == W and seed.shape[2] == len(chars) )


print 'Generating 2000 chars'
for i in range(2000):
    if i%100==0:
        print 'Generated char #',i
    pr = model.predict( seed )[0].argmax() #index of prediction.
    pr_ch = indices_char[ pr ] #look this up in the vocab

    # Store/print the prediction
    # print '%c' %(pr_ch),
    ALL_GEN.append( pr_ch )
    # print pr_ch, #' (', pr, ')'

    # Make a new seed using the prediction in this current step,
    new_seed = np.zeros( seed.shape )
    new_seed[0,0:-1,:] = seed[0,1:,:]
    new_seed[0,-1,pr] = 1.

    seed = new_seed

print ''.join( ALL_GEN )

print ''
print ''.join( [ indices_char[v] for v in orf_seed.argmax( axis=-1 )[0] ] )



# After learning this net for about 120 epochs (took an hour on my TitanX gpu.)
# here, is a sample generation:
#

"""
ess cannition and most cricific man in every painity, even in compleceiring the ene is not one sunses the heart in deemine in fact the historical process only and is alrays pastical, and sension of the such always passible accisting in the senses of the here, the german sunses the highest pan of the his eximent, the higher sense of the highest and string is the
presence, of his exile the same might prisic still be
it seess in it in europe of the himself of the here, himself string to these sense of their period, should reficed in the free, spirit and superstick the ene, that is the higher may and in disinging and priciss and things and shill of his exile respires the highest pancis of the exill, in the higher man in every something eximited in their condition of man is an action as and
socratic, even in the higher still in their persent spiriture of things and sefulity
is approposition to the instinct and sifceence than the highest and self-sonthing the
highest and self-sontinermy in the revent provers of should concealing, an instinct of the highest indistoration and self-men of the such altored to the struggle is preception to the free spirit and superstick the sense of the here, himself and is something in the higher man in even in the highest man carefuling in the free spirit and such the here is there is not be in adericated in the freements of strent and suppression, shill wish there is a peesent in it, in europe of the highest man cannot at the extent of gher in the secret compretion, he is thereby alt the higher man and something experiences of the exil, the free spirit and supers in the highest man an ansthing example, in the has dired in it, in every rocerfing it, and in it not present it is a persent of the highest man an ansting is a personsion of the here, the
spirit and self-expresion of the instinct, and essivition
of things, the
preperion of the exilling and self-decisive men is a possibies this means of hears and self-some in the reveric of his elis

"""


#Another one:
"""
nssm, and brought of every noble me, of things exercised, be even to be all respectf of the persons, as the costinn of the besists (houths!" on which on higher we look of formand the fact that the heart of stiented to hoso becrees whether
one pain as something formert--and objected by the goads of demands accests be. af it, as denencerre--the one middaling exoliticaling be fail, fellowed to sceet distrust of the cleadly be the pass of dissessible belong.--this is lessives about the beso,. when at epistees. beean the besole, be their loat. on the last
reference they have been atterment of an amont the lostical deseptives, becomes theue life.


22

=pihio ofree spirit previves become thing---
   peetive of his egeracy and donounality; only what every
possibles befuned, the prople of the perspetion, very veing onesess of will
the lowar asternal heals all this myself, that the costinupod, at lase oppopetiness, the astsibity of thing--(leckuped be the perilacting persentive and this disy-sing inconcience that principle morality, and opposes
whece at present perhaps on the cosess behomest his syowed, has libere oppoose--which has hitherto has no laties and been
amples be taste, as the possibition to the best,
-werl, will be the discovery of the very opscasore even with the egerness
and among become heads only be appeared even to the best of which is afforded be every
doubtly enere--when we werk loves been from the love, fee
ropeaife, even to itself heropent in every nablement of the forlies. in fley leadnt of entential enouse be every voltucy of one's sexials to prese--a pore we part of human is sometes, by every very
compresencing of his so called be no value
to be other, of fatually, eare for a compone, by mast respect for his gere all the beso sained. an abys the great nextranct of the existence be the value of life, the acts for his
sons--fur the himself of his exile, becent and distrustfelt fear and invossesses of
epechaps, that estives, perhaps, profiences of "man"-
"""
