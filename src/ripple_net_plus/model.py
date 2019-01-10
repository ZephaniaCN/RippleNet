import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score


class RippleNetPlus(object):
    def __init__(self, args, n_entity, n_relation):
        self._parse_args(args, n_entity, n_relation)
        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops
        self.embed_size= args.embed_size
        self.batch_size = args.batch_size

    def _build_inputs(self):
        with tf.variable_scope('input'):
            self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
            self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")
            self.memories_h = []
            self.memories_r = []
            self.memories_t = []

            for hop in range(self.n_hop):
                self.memories_h.append(
                    tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
                self.memories_r.append(
                    tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
                self.memories_t.append(
                    tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))

    def _build_embeddings(self):
        with tf.variable_scope('input'):
            self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float64,
                                                     shape=[self.n_entity, self.dim],
                                                     initializer=tf.contrib.layers.xavier_initializer())
            self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float64,
                                                       shape=[self.n_relation, self.dim, self.dim],
                                                       initializer=tf.contrib.layers.xavier_initializer())

    def _build_model(self):
        # transformation matrix for updating item embeddings at the end of each hop
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float64,
                                                initializer=tf.contrib.layers.xavier_initializer())

        # [batch size, dim]
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))

            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r[i]))

            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        o = self.eposid_memory()

        self.scores = tf.squeeze(self.predict(self.item_embeddings, o))
        self.scores_normalized = tf.sigmoid(self.scores)

    def eposid_memory(self):
        # [batch_size, dim, 1]
        with tf.variable_scope('eposid_memory'):
            #[batch_size, dim]
            v = self.item_embeddings
            prev_memory = tf.identity(v)
            for hop in range(self.n_hop):
                # [batch_size, n_memory, dim, 1]
                h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

                # [batch_size, n_memory, dim]
                Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

                t = self.t_emb_list[hop]
                # [batch_size, dim]
                o = self.attention_gate(v,prev_memory,Rh,t)

                o, prev_memory = self.update_memory(o,prev_memory)

            return o


    def attention_gate(self,v,prev_memory,Rh,t):
        with tf.variable_scope("attention",reuse=tf.AUTO_REUSE):
            attentions = []
            for i in range(self.n_memory):
                attentions.append(self.get_attention(v, prev_memory, Rh[:, i, :]))

            attentions = tf.transpose(tf.squeeze(tf.stack(attentions)))
            attentions = tf.expand_dims(tf.nn.softmax(attentions), axis=2)
            o = tf.reduce_sum(t * attentions, axis=1)
            o.set_shape([None, self.dim])
            return o

    def update_memory(self,o,prev_memory):
        with tf.variable_scope("update_memory",reuse=tf.AUTO_REUSE):
            self.rnn = tf.contrib.rnn.GRUCell(self.dim)
            output, update_memory = self.rnn(o, prev_memory)
            return output, update_memory

    def get_attention(self, q_vec, prev_memory, fact_vec):
        with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
            # build feature vector
            features = [fact_vec * q_vec,
                        fact_vec * prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            # two layer nn
            attention = tf.contrib.layers.fully_connected(feature_vec,
                                                          self.embed_size,
                                                          activation_fn=tf.nn.tanh,
                                                          reuse=tf.AUTO_REUSE, scope="fc1")

            attention = tf.contrib.layers.fully_connected(attention,
                                                          1,
                                                          activation_fn=None,
                                                          reuse=tf.AUTO_REUSE, scope="fc2")
            return attention

    def predict(self, item_embeddings, o):
        with tf.variable_scope('predict'):
        # [batch_size]
            scores = tf.reduce_sum(item_embeddings * o, axis=1)
            return scores

    def _build_loss(self):
        with tf.variable_scope('loss'):
            self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

            self.kge_loss = 0
            for hop in range(self.n_hop):
                h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
                t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
                hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
                self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
            self.kge_loss = -self.kge_weight * self.kge_loss

            self.l2_loss = 0
            for hop in range(self.n_hop):
                self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
                self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
                self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
                if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                    self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
            self.l2_loss = self.l2_weight * self.l2_loss

            self.loss = self.base_loss + self.kge_loss + self.l2_loss

    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, clip_norm=5)
                     for gradient in gradients]
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables))
        '''

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc
