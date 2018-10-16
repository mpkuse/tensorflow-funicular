# Tensorflow Examples

View my blog posts on neural networks : [https://kusemanohar.wordpress.com/tag/neural-network/](https://kusemanohar.wordpress.com/tag/neural-network/).

Here I am consolidating and upgrading my code to be compatible with tf1.10. In particular, I
am moving all the network construction part to keras. This is just my
tensorflow/keras basic examples.

## Examples

#### Fully Connected Neural Nets
- [1_iris.py](1_iris.py): 4 input features to predict amongst 3 classes. Build a 3-layer neural network and learn to predict.
- [3_mnist_mlp.py](3_mnist_mlp.py): Fit a 3 layer fully connect network on mnist dataset. For classifying an image (reshaped to a vector) into 1 of 10 classes.

#### Convolutional Nets
- [3_mnist_cnn.py](3_mnist_cnn.py): Fit a 4-conv layer and 2 fully-connected layer to classify digits on the mnist dataset
- [4_cifar10_cnn.py](4_cifar10_cnn.py): A conv net on cifar10 dataset. It is 10 category classification in natural images.

#### Keras Functionalities
- [4_cifar10_customloss.py](4_cifar10_customloss.py): How to implement (and use) a custom loss function for training.
- [4_cifar10_functionalmodelapi.py](4_cifar10_functionalmodelapi.py): Building complicated neural models with keras's functional api.
- [6_keras_applications.py](6_keras_applications.py): Using pretrained nets for VGG16, ResNet, Xception, Inception etc. through keras.applications

#### Recurrent Networks
- [7_rnn.py](7_rnn.py): Toy RNN example from [Karpaty's blog post on rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- [7_rnn_on_text.py](7_rnn_on_text.py): Learning to predict next char with RNN.
Except for the fact that I use
an LSTM (instead of RNN. LSTM is a kind of RNN). A fantastic [explaination on LSTM by colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

#### Misc
- [5_fcn.py](5_fcn.py): Exploring transposed-convolution (upsampling conv). 

## References
- [https://github.com/Hvass-Labs/TensorFlow-Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
- [https://keras.io/](https://keras.io/)
- [https://github.com/keras-team/keras/tree/master/examples](https://github.com/keras-team/keras/tree/master/examples)
- [https://www.tensorflow.org/api_docs/python/tf](https://www.tensorflow.org/api_docs/python/tf)

## How I learned to use Tensorflow
I already had a theoritical understanding of neural networks. I think, the most authentic source is
[Stanford's cs231n](http://cs231n.stanford.edu/). I just watched all the lectures and implemented
a crude neural network from absolute scratch.

After I had the basic, I started reading papers. There are several lists on the internet
which maintain a list of must read deep learning papers. See [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers) for example.

After all this, I got my hands dirty with tensorflow. I started with tf0.6. Tensorflow is continuously
evolving so it is important to read the documentation and/or look at official examples. I think this is best
way to get familiar with standard tricks. Top-end folks out there don't have time to write up
tutorials. You might also have a look at pytorch, it is a pretty neat tool to fireup your neural nets.  
Although I still stick to tensorflow.

Current recommended way is to use keras with tensorflow. Keras defines a pretty intuitive API.
Although a new user may have a very steep learning curve. I followed the tensorflow tutorial which
are too basic (does not expose to lot of real cases tensorflow can be used). After that, can
move onto keras examples.
