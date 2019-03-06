import tensorflow as tf

print 'You seem to have tensorflow version = ', tf.__version__ 
print 'Creates a graph.'
print('a')
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
print('b')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
print('c')
c = tf.matmul(a, b)
print('d')

print 'Creates a session with log_device_placement set to True.'
print('e')

sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))
## Runs the op.
print 'Run the Op' 
print('f')
print(sess.run(c))
