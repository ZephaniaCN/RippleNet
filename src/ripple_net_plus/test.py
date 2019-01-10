import tensorflow as tf
import numpy as np
n_memory = 3
dim = 2
embed_size=80
batch_size=4
num_units = dim

Rh = tf.placeholder(tf.float32,[None,n_memory,dim])
v = tf.placeholder(tf.float32, [None,dim])
t = tf.placeholder(tf.float32, [None,n_memory,dim])
# attetion

rnn = tf.contrib.rnn.GRUCell(num_units)

def get_attention(q_vec, prev_memory, fact_vec):
    with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
        # build feature vector
        features = [fact_vec * q_vec,
                    fact_vec * prev_memory,
                    tf.abs(fact_vec - q_vec),
                    tf.abs(fact_vec - prev_memory)]

        feature_vec = tf.concat(features, 1)

        # two layer nn
        attention = tf.contrib.layers.fully_connected(feature_vec,
                                                      embed_size,
                                                      activation_fn=tf.nn.tanh,
                                                      reuse=tf.AUTO_REUSE, scope="fc1")

        attention = tf.contrib.layers.fully_connected(attention,
                                                      1,
                                                      activation_fn=None,
                                                      reuse=tf.AUTO_REUSE, scope="fc2")
        return attention

prev_memory = tf.identity(v)

attentions = []
for i in range(n_memory):
    attentions.append(get_attention(v, prev_memory, Rh[:,i,:]))

attentions = tf.transpose(tf.squeeze(tf.stack(attentions)))
attentions = tf.expand_dims(tf.nn.softmax(attentions),axis=2)
o  = tf.reduce_sum(t*attentions,axis=1)
o.set_shape([None,dim])
_,prev_memory = rnn(o,prev_memory)

def fake_date():
    return {Rh:np.random.randn(batch_size,n_memory,dim),v:np.random.randn(batch_size,dim),t:np.random.randn(batch_size,n_memory,dim)}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    prev_memory,v,o=sess.run([prev_memory,v,o],fake_date())

    print("prev_memory:{}\nv:{}\no:{}".format(prev_memory,v,o))