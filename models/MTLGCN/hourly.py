# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
import collections

GCLSTMCell = collections.namedtuple("GCLSTMCell", ('c','h'))

class hourlyCell(RNNCell):
    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, adj, num_nodes, w_att,order_k,
                 act=tf.nn.tanh, reuse=None):

        super(hourlyCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nNode = num_nodes
        self._units = num_units
        self._adj = adj
        self.K = order_k
        self.w_att = w_att


    @property
    def state_size(self):
        return (GCLSTMCell((self._nNode, self._units),(self._nNode, self._units)))

    @property
    def output_size(self):
        return self._units
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + 'myZeroState'): 
            zero_state_c = tf.zeros([batch_size, self._nNode, self._units], name='c')
            zero_state_h = tf.zeros([batch_size, self._nNode, self._units], name='h')
            return (zero_state_c, zero_state_h)


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state
            f_out = self._units

            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                stddev = 0.01

                Theta1 = []
                for idx in range(self.K):
                    Theta1.append(tf.get_variable(name="theta1_" + str(idx), shape=[1, f_out],
                                                  initializer=tf.random_normal_initializer(stddev=0.01)))

                x = self.gcn(inputs, self._adj, self.w_att, f_out, self.K, Theta1)

                w_ixt = tf.get_variable("w_ixt", [f_out, f_out], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=stddev))
                w_fxt = tf.get_variable("w_fxt", [f_out, f_out], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=stddev))
                w_oxt = tf.get_variable("w_oxt", [f_out, f_out], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=stddev))
                w_zxt = tf.get_variable("w_zxt", [f_out, f_out], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=stddev))

                w_iht = tf.get_variable("w_iht", [f_out, f_out], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=stddev))
                w_fht = tf.get_variable("w_fht", [f_out, f_out], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=stddev))
                w_oht = tf.get_variable("w_oht", [f_out, f_out], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=stddev))
                w_zht = tf.get_variable("w_zht", [f_out, f_out], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=stddev))

                bias_zt = tf.get_variable("bias_zt", [f_out])
                bias_it = tf.get_variable("bias_it", [f_out])
                bias_ft = tf.get_variable("bias_ft", [f_out])
                bias_ot = tf.get_variable("bias_ot", [f_out])


                ixt = tf.reshape(tf.matmul(tf.reshape(x, [-1, f_out]), w_ixt), [-1, self._nNode, f_out])
                iht = tf.reshape(tf.matmul(tf.reshape(h, [-1, f_out]), w_iht), [-1, self._nNode, f_out])
                it = ixt + iht + bias_it
                it = tf.sigmoid(it)

                fxt = tf.reshape(tf.matmul(tf.reshape(x, [-1, f_out]), w_fxt), [-1, self._nNode, f_out])
                fht = tf.reshape(tf.matmul(tf.reshape(h, [-1, f_out]), w_fht), [-1, self._nNode, f_out])
                ft = fxt + fht + bias_ft
                ft = tf.sigmoid(ft)

                oxt = tf.reshape(tf.matmul(tf.reshape(x, [-1, f_out]), w_oxt), [-1, self._nNode, f_out])
                oht = tf.reshape(tf.matmul(tf.reshape(h, [-1, f_out]), w_oht), [-1, self._nNode, f_out])
                ot = oxt + oht + bias_ot
                ot = tf.sigmoid(ot)

                zxt = tf.reshape(tf.matmul(tf.reshape(x, [-1, f_out]), w_zxt), [-1, self._nNode, f_out])
                zht = tf.reshape(tf.matmul(tf.reshape(h, [-1, f_out]), w_zht), [-1, self._nNode, f_out])
                zt = zxt + zht + bias_zt
                zt = tf.tanh(zt)

                new_c = ft * c + it * zt
                new_h = ot * tf.tanh(new_c)

                new_state = GCLSTMCell(new_c, new_h)
                return new_h, new_state

    def gcn(self, x, adj, w_att, out_channels, K, Theta):
        _, nNode, in_channels = x.get_shape().as_list()
        output = tf.zeros([tf.shape(x)[0],nNode,out_channels])
        for k in range(K):
            if k==0:
                rhs = x
            else:
                rhs = tf.matmul(tf.reshape(tf.transpose(x, [0,2,1]),[-1,nNode]), adj[k-1]*w_att)
                rhs = tf.reshape(rhs,[-1,in_channels,nNode])
                rhs = tf.transpose(rhs, [0,2,1])
            rhs = tf.reshape(rhs, [-1,in_channels])
            output = output + tf.reshape(tf.matmul(rhs, Theta[k]),[-1,nNode,out_channels])
        return output



