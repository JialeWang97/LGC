import tensorflow as tf
import numpy as np
import os
from data import *
from weekly import weeklyCell
from hourly import hourlyCell
from daily import dailyCell
import time
import sys

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('Epoch', 200, 'Number of epochs to train.')
flags.DEFINE_integer('Hidden_units', 64, 'Hidden units of lstm and GCN.')
flags.DEFINE_integer('Seq_len',12, '  Time length of inputs.')
flags.DEFINE_integer('Pre_len', 12, 'Time length of prediction.')
flags.DEFINE_integer('Batch_size', 50, 'batch size.')
flags.DEFINE_integer('Order', 3, 'Graph convolution order k')
flags.DEFINE_string('Dataset', 'BJ-speed', 'Dataset name')
flags.DEFINE_boolean('Self_learning', True, 'MTLGCN or MTGCN')
flags.DEFINE_integer('Num_nodes', 801, 'The number of nodes')
flags.DEFINE_integer('Feature_dim', 1, 'The dimensionality of input vector')
flags.DEFINE_string('Date_list_path', r'..\..\data\Urban\date_list.csv', 'Date list path')

class Train(object):
    def __init__(self, config):
        self.epoch = config.Epoch
        self.hidden_units = config.Hidden_units
        self.seq_len = config.Seq_len
        self.pre_len = config.Pre_len
        self.batch_size = config.Batch_size
        self.order = config.Order
        self.dataset = config.Dataset
        self.self_learning = config.Self_learning
        self.num_nodes = config.Num_nodes
        self.date = config.Date_list_path
        self.dim_in = config.Feature_dim

        # Load dataset
        if self.dataset == "Urban":
            spd, adj1, adj2, adj3, adj4, adj5 = load_urban_speed_data()
            self.data = spd
            train_set, valid_set, test_set, mean, std = process(spd, self.date, self.pre_len)
        if self.dataset == "PeMSD8":
            volume, adj1, adj2, adj3, adj4, adj5 = load_pems08_volume_data()
            train_set, valid_set, test_set, mean, std = process(volume, self.date, self.pre_len)
            self.data = volume

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.mean = mean
        self.std = std
        self.adj = [adj1, adj2, adj3, adj4, adj5]

        self.train_total_batch = int(train_set.shape[1] / self.batch_size) + 1
        self.test_total_batch = int(test_set.shape[1] / self.batch_size) + 1
        self.valid_total_batch = int(valid_set.shape[1] / self.batch_size) + 1

    def bulid(self):
        # Model inputs
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.num_nodes, self.dim_in])
        self.inputs1 = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.num_nodes, self.dim_in])
        self.inputs2 = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.num_nodes, self.dim_in])
        self.labels = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.num_nodes, self.dim_in])

        w_att = tf.get_variable("w_att", shape=[self.num_nodes, self.num_nodes], dtype=tf.float32,
                                initializer=tf.constant_initializer(1))

        week_cell = weeklyCell(self.hidden_units, self.adj, w_att=w_att, num_nodes=self.num_nodes, order_k=self.order + 1)
        _X = tf.unstack(self.inputs, axis=1)
        week_outputs, week_hidden = tf.nn.static_rnn(week_cell, _X, dtype=tf.float32)

        day_cell = dailyCell(self.hidden_units, self.adj, w_att=w_att, num_nodes=self.num_nodes, order_k=self.order + 1)
        _X1 = tf.unstack(self.inputs1, axis=1)
        day_outputs, day_hidden = tf.nn.static_rnn(day_cell, _X1, dtype=tf.float32)

        hour_cell = hourlyCell(self.hidden_units, self.adj, w_att=w_att, num_nodes=self.num_nodes, order_k=self.order + 1)
        _X2 = tf.unstack(self.inputs2, axis=1)
        hour_outputs, hour_hidden = tf.nn.static_rnn(hour_cell, _X2, dtype=tf.float32)

        week_outputs = tf.transpose(week_outputs, [1, 0, 2, 3])
        day_outputs = tf.transpose(day_outputs, [1, 0, 2, 3])
        hour_outputs = tf.transpose(hour_outputs, [1, 0, 2, 3])

        w_cnn_w = tf.get_variable('cnn1_w', shape=[self.seq_len, 1, self.hidden_units, self.hidden_units], dtype=tf.float32)
        w_cnn2_d = tf.get_variable('cnn2_d', shape=[self.seq_len, 1, self.hidden_units, self.hidden_units], dtype=tf.float32)
        w_cnn3_h = tf.get_variable('cnn3_h', shape=[self.seq_len, 1, self.hidden_units, self.hidden_units], dtype=tf.float32)
        # => batch, 1, nNode, channels
        conv_week_out = tf.nn.conv2d(week_outputs, w_cnn_w, strides=[1, 1, 1, 1], padding='VALID')
        conv_day_out = tf.nn.conv2d(day_outputs, w_cnn2_d, strides=[1, 1, 1, 1], padding='VALID')
        conv_hour_out = tf.nn.conv2d(hour_outputs, w_cnn3_h, strides=[1, 1, 1, 1], padding='VALID')

        w_cnn_w1 = tf.get_variable('cnn1_w1', shape=[1, 1, self.hidden_units, self.pre_len], dtype=tf.float32)
        w_cnn2_d1 = tf.get_variable('cnn2_d1', shape=[1, 1, self.hidden_units, self.pre_len], dtype=tf.float32)
        w_cnn3_h1 = tf.get_variable('cnn3_h1', shape=[1, 1, self.hidden_units, self.pre_len], dtype=tf.float32)
        ##batch 1 nNode 12
        conv_week_out = tf.nn.conv2d(conv_week_out, w_cnn_w1, strides=[1, 1, 1, 1], padding='SAME')
        conv_day_out = tf.nn.conv2d(conv_day_out, w_cnn2_d1, strides=[1, 1, 1, 1], padding='SAME')
        conv_hour_out = tf.nn.conv2d(conv_hour_out, w_cnn3_h1, strides=[1, 1, 1, 1], padding='SAME')

        y_pred_week = tf.reshape(conv_week_out, [-1, self.num_nodes, self.pre_len])
        y_pred_day = tf.reshape(conv_day_out, [-1, self.num_nodes, self.pre_len])
        y_pred_hour = tf.reshape(conv_hour_out, [-1, self.num_nodes, self.pre_len])

        y_pred = y_pred_week + y_pred_day + y_pred_hour  # batch,nNode,12

        self.y_pred = tf.transpose(y_pred, [0, 2, 1])

        ###### optimizer ######
        _label = tf.reshape(self.labels, [-1, self.num_nodes])
        _y_pred = tf.reshape(self.y_pred, [-1, self.num_nodes])
        _loss = tf.square(_y_pred - _label)
        self.loss = tf.reduce_mean(_loss)
        ##rmse
        self.error = tf.sqrt(self.loss)
        self.global_steps = tf.Variable(0, trainable=False)
        if self.dataset=='BJ-speed':
            lr = 0.001
        if self.dataset=='PeMSD8':
            lr = tf.train.exponential_decay(0.0015, self.global_steps, decay_steps=3, decay_rate=0.9, staircase=True)

        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self):
        time_start = time.time()

        variables = tf.global_variables()
        saver = tf.train.Saver(tf.global_variables())

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        batch_loss, batch_rmse, batch_pred = [], [], []
        test_loss, test_rmse, test_mae, test_mape, test_pred = [], [], [], [], []

        min_valid_loss = sys.float_info.max
        min_test_rmse = sys.float_info.max

        for epoch in range(self.epoch):
            print("epoch", epoch)
            for m in range(self.train_total_batch):
                mini_batch = self.train_set[0][m * self.batch_size: (m + 1) * self.batch_size]
                mini_batch1 = self.train_set[1][m * self.batch_size: (m + 1) * self.batch_size]
                mini_batch2 = self.train_set[2][m * self.batch_size: (m + 1) * self.batch_size]
                mini_label = self.train_set[3][m * self.batch_size: (m + 1) * self.batch_size]

                _, loss1, rmse1, train_output = sess.run([self.optimizer, self.loss, self.error, self.y_pred],
                                                         feed_dict={self.inputs: mini_batch, self.inputs1: mini_batch1,
                                                                    self.inputs2: mini_batch2, self.labels: mini_label,
                                                                    self.global_steps: epoch})
                batch_loss.append(loss1)
                batch_rmse.append(rmse1)

            valid_LOSS = 0
            for tb in range(self.test_total_batch):
                valid_mini_batch = self.valid_set[0][tb * self.batch_size: (tb + 1) * self.batch_size]
                valid_mini_batch1 = self.valid_set[1][tb * self.batch_size: (tb + 1) * self.batch_size]
                valid_mini_batch2 = self.valid_set[2][tb * self.batch_size: (tb + 1) * self.batch_size]
                valid_mini_label = self.valid_set[3][tb * self.batch_size: (tb + 1) * self.batch_size]
                _loss_valid = sess.run(self.loss, feed_dict={self.inputs: valid_mini_batch, self.inputs1: valid_mini_batch1,
                                                             self.inputs2: valid_mini_batch2, self.labels: valid_mini_label,
                                                             self.global_steps: epoch})
                valid_LOSS += _loss_valid

            if min_valid_loss > valid_LOSS:
                min_valid_loss = valid_LOSS

                test_output = []
                test_LOSS = 0
                for tb in range(self.valid_total_batch):
                    test_mini_batch = self.test_set[0][tb * self.batch_size: (tb + 1) * self.batch_size]
                    test_mini_batch1 = self.test_set[1][tb * self.batch_size: (tb + 1) * self.batch_size]
                    test_mini_batch2 = self.test_set[2][tb * self.batch_size: (tb + 1) * self.batch_size]
                    test_mini_label = self.test_set[3][tb * self.batch_size: (tb + 1) * self.batch_size]
                    test_batch_output, _loss_test = sess.run([self.y_pred, self.loss],
                                                             feed_dict={self.inputs: test_mini_batch, self.inputs1: test_mini_batch1,
                                                                        self.inputs2: test_mini_batch2, self.labels: test_mini_label,
                                                                        self.global_steps: epoch})
                    test_LOSS += _loss_test

                    if tb == 0:
                        test_output = test_batch_output

                    else:
                        test_output = np.concatenate([test_output, test_batch_output], axis=0)

                test_label = np.reshape(self.test_set[3], [-1, self.num_nodes])
                rmse, mae, mape = self.evaluation(test_label, test_output)

                test_output1 = (test_output * self.std) + self.mean
                test_loss.append(test_LOSS)
                test_rmse.append(rmse)
                test_mae.append(mae)
                test_mape.append(mape)

                if rmse < min_test_rmse:
                    min_test_rmse = rmse
                    test_pred = test_output1

                print('Iter:{}'.format(epoch),
                      'train_rmse:{:.4}'.format(batch_rmse[-1]),
                      'test_loss:{:.4}'.format(test_LOSS),
                      'test_rmse:{:.4}'.format(test_rmse[-1]),
                      'test_mape:{:.4}'.format(mape),
                      'test_mae:{:.4}'.format(mae),
                      'error_rmse:{:.4}'.format(rmse))

        time_end = time.time()
        print(time_end - time_start, 's')

        b = int(len(batch_rmse) / self.train_total_batch)
        batch_rmse1 = [i for i in batch_rmse]
        train_rmse = [(sum(batch_rmse1[i * self.train_total_batch:(i + 1) * self.train_total_batch]) / self.train_total_batch) for i in range(b)]
        batch_loss1 = [i for i in batch_loss]
        train_loss = [(sum(batch_loss1[i * self.train_total_batch:(i + 1) * self.train_total_batch]) / self.train_total_batch) for i in range(b)]

        return test_pred,train_rmse, train_loss, test_rmse, test_mape, test_mae

    def MAPE(self, v, v_):
        mape = 0
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if abs(v[i][j] > 1e-3):
                    mape += np.abs(v_[i][j] - v[i][j]) / (v[i][j])
        return mape / v.shape[0] / v.shape[1]

    def RMSE(self, v, v_):
        return np.sqrt(np.mean((v_ - v) ** 2))

    def MAE(self, v, v_):
        return np.mean(np.abs(v_ - v))

    def evaluation(self, label1, pred_out):
        pred = np.reshape(pred_out, [-1, self.num_nodes])
        pred = (pred * self.std) + self.mean
        label = (label1 * self.std) + self.mean
        return self.RMSE(label, pred), self.MAE(label, pred), self.MAPE(label, pred)
