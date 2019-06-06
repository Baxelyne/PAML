# -*- coding: utf-8 -*-
# @FileName: PAMLModel.py

import numpy as np
import time, os
import pandas as pd
import tensorflow as tf
import math
from engines.utils import metrics, save_csv_

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PAML:
    def __init__(self, configs, logger, dataManager):
        self.configs = configs
        self.logger = logger
        self.logdir = configs.log_dir
        self.measuring_metrics = configs.measuring_metrics
        self.dataManager = dataManager

        if configs.mode == "train":
            self.is_training = True
        else:
            self.is_training = False
        self.checkpoint_name = configs.checkpoint_name
        self.checkpoints_dir = configs.checkpoints_dir
        self.output_test_file = configs.datasets_fold + "/" + configs.output_test_file
        self.is_output_sentence_entity = configs.is_output_sentence_entity
        self.output_sentence_entity_file = configs.datasets_fold + "/" + configs.output_sentence_entity_file

        self.learning_rate = configs.learning_rate
        self.dropout_rate = configs.dropout
        self.batch_size = configs.batch_size

        self.emb_dim = configs.embedding_dim
        self.lstm_embedding_dims = configs.hidden_dim

        self.is_attention = configs.use_self_attention
        self.attention_dim = configs.attention_dim

        self.num_epochs = configs.epoch
        self.lstm_max_step = configs.max_sequence_length

        self.num_tokens = dataManager.max_token_number
        self.output_dim = dataManager.max_label_number

        self.is_early_stop = configs.is_early_stop
        self.patient = configs.patient

        self.max_to_keep = configs.checkpoints_max_to_keep
        self.print_per_batch = configs.print_per_batch
        self.best_f1_val = 0

        self.position_size = configs.position_size
        self.position_dim = configs.position_dim

        self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)
        self.initializer = tf.contrib.layers.xavier_initializer()

        if configs.use_pretrained_embedding:
            embedding_matrix = dataManager.getEmbedding(configs.token_emb_dir)
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.num_tokens, self.emb_dim], trainable=True,
                                             initializer=self.initializer)

        self._build_palceholder()
        # basic underlying network
        self._build_underlying_network()
        # multi-task
        self._build_task_networks()

        # loss
        self.loss_T = 0.5 * self.loss_task1 + 0.3 * self.loss_task2 + 0.2 * self.loss_task3
        self.loss = self.loss_T

        self.opt_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def positional_encoding(self, dtype=tf.int32):
        encoded_vec = np.array([pos for pos in range(self.max_to_keep)])
        return tf.convert_to_tensor(encoded_vec, dtype=dtype)

    def _build_palceholder(self):
        self.position_wv1 = tf.get_variable("position_wv1", [self.position_size, self.position_dim], trainable=True,
                                            initializer=self.initializer)
        self.position_wv2 = tf.get_variable("position_wv2", [self.position_size, self.position_dim], trainable=True,
                                            initializer=self.initializer)
        self.position_wv3 = tf.get_variable("position_wv3", [self.position_size, self.position_dim], trainable=True,
                                            initializer=self.initializer)

        self.text_input = tf.placeholder(shape=[None, self.lstm_max_step], dtype=tf.int32, name="input")

        self.length = tf.cast(tf.reduce_sum(tf.sign(self.text_input), reduction_indices=1), tf.int32)

        self.output_task1 = tf.placeholder(shape=[None, self.lstm_max_step], dtype=tf.int32, name="output_task1")
        self.position_task1_input = self.positional_encoding()
        self.position_task1_wv = tf.nn.embedding_lookup(self.position_wv1, self.position_task1_input,
                                                        name="position_task1_wv")

        self.output_task2 = tf.placeholder(shape=[None, self.lstm_max_step], dtype=tf.int32, name="output_task2")
        self.position_task2_input = self.positional_encoding()
        self.position_task2_wv = tf.nn.embedding_lookup(self.position_wv2, self.position_task2_input,
                                                        name="position_task2_wv")

        self.output_task3 = tf.placeholder(shape=[None, self.lstm_max_step], dtype=tf.int32, name="output_task3")
        self.position_task3_input = self.positional_encoding()
        self.position_task3_wv = tf.nn.embedding_lookup(self.position_wv3, self.position_task3_input,
                                                        name="position_task3_wv")

    def kl_divergence(self, p, q):
        '''
        This assumes that p and q are both 1-D tensors of floats, of the same shape and for each their values sum to 1.
        '''
        return tf.reduce_sum(p * tf.log(p / q))

    def _build_underlying_network(self):

        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.text_input)
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
        self.inputs_emb = tf.split(self.inputs_emb, self.lstm_embedding_dims, 0)

        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.lstm_embedding_dims, initializer=self.initializer,
                                               state_is_tuple=False)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.lstm_embedding_dims, initializer=self.initializer,
                                               state_is_tuple=False)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
            lstm_cell_fw,
            lstm_cell_bw,
            self.inputs_emb,
            dtype=tf.float32,
            sequence_length=self.length
        )
        self.repre_LSTM = tf.concat(outputs, 1)  # [batch, steps, 2*dim]
        self.repre_LSTM = tf.reshape(self.repre_LSTM,
                                     [self.batch_size, self.lstm_max_step, self.lstm_embedding_dims * 2])

    def _build_task_networks(self):
        # position-aware attention
        self._build_task1()
        self._build_task2()
        self._build_task3()

    def _build_task1(self):

        med_outputs1 = self.repre_LSTM

        with tf.name_scope("task1"):
            ### attention
            if self.is_attention:
                H = tf.reshape(med_outputs1, [-1, self.lstm_embedding_dims * 2])
                W_a = tf.get_variable("W_a_H1", shape=[self.lstm_embedding_dims * 2, self.attention_dim],
                                      initializer=self.initializer, trainable=True)
                H_u = tf.reshape(tf.matmul(H, W_a), [self.batch_size, self.lstm_max_step, self.lstm_embedding_dims * 2])

                position_H = tf.reshape(self.position_task1_wv, [-1, self.position_dim])
                W_a_p = tf.get_variable("W_a_pos1", shape=[self.position_dim, self.attention_dim],
                                        initializer=self.initializer, trainable=True)
                H_p = tf.reshape(tf.matmul(position_H, W_a_p),
                                 [self.batch_size, self.lstm_max_step, self.lstm_embedding_dims * 2])

                u = tf.matmul(H_u, H_p, transpose_b=True)
                self.alpha_task1 = tf.nn.softmax(u)
                self.attention_context_1 = tf.matmul(self.alpha_task1, med_outputs1)
                med_outputs1 = self.attention_context_1

            # linear
            med_outputs1 = tf.reshape(med_outputs1, [-1, self.lstm_embedding_dims * 2])
            self.softmax_w1 = tf.get_variable("softmax_w1", [self.lstm_embedding_dims * 2, self.output_dim],
                                              initializer=self.initializer)
            self.softmax_b1 = tf.get_variable("softmax_b1", [self.output_dim], initializer=self.initializer)
            self.logits1 = tf.matmul(med_outputs1, self.softmax_w1) + self.softmax_b1

            # crf
            self.tags_scores1 = tf.reshape(self.logits1, [self.batch_size, self.lstm_max_step, self.output_dim])
            self.log_likelihood1, self.transition_params1 = tf.contrib.crf.crf_log_likelihood(
                self.tags_scores1, self.output_task1, self.length, name='task1')
            self.batch_pred_sequence1, self.batch_viterbi_score1 = tf.contrib.crf.crf_decode(self.tags_scores1,
                                                                                             self.transition_params1,
                                                                                             self.length)

            self.loss_task1 = tf.reduce_mean(-self.log_likelihood1)

    def _build_task2(self):
        # attention_task2
        med_outputs2 = self.repre_LSTM

        with tf.name_scope("task2"):
            ### attention
            if self.is_attention:
                H = tf.reshape(med_outputs2, [-1, self.lstm_embedding_dims * 2])
                W_a = tf.get_variable("W_a_H2", shape=[self.lstm_embedding_dims * 2, self.attention_dim],
                                      initializer=self.initializer, trainable=True)
                H_u = tf.reshape(tf.matmul(H, W_a), [self.batch_size, self.lstm_max_step, self.lstm_embedding_dims * 2])

                position_H = tf.reshape(self.position_task2_wv, [-1, self.position_dim])
                W_a_p = tf.get_variable("W_a_pos2", shape=[self.position_dim, self.attention_dim],
                                        initializer=self.initializer, trainable=True)
                H_p = tf.reshape(tf.matmul(position_H, W_a_p),
                                 [self.batch_size, self.lstm_max_step, self.lstm_embedding_dims * 2])

                u = tf.matmul(H_u, H_p, transpose_b=True)
                self.alpha_task2 = tf.nn.softmax(u)
                context_2 = tf.matmul(self.alpha_task2, med_outputs2)
                self.attention_context_2 = tf.concat([context_2, self.attention_context_1], -1)
                self.attention_context_2 = tf.reshape(
                    tf.layers.dense(self.attention_context_2, self.lstm_embedding_dims * 2, name="linear2",
                                    use_bias=False),
                    [self.batch_size, self.lstm_max_step, self.lstm_embedding_dims * 2])
                med_outputs2 = self.attention_context_2

            med_outputs2 = tf.reshape(med_outputs2, [-1, self.lstm_embedding_dims * 2])
            self.softmax_w2 = tf.get_variable("softmax_w2", [self.lstm_embedding_dims * 2, self.output_dim],
                                              initializer=self.initializer)
            self.softmax_b2 = tf.get_variable("softmax_b2", [self.output_dim], initializer=self.initializer)
            self.logits2 = tf.matmul(med_outputs2, self.softmax_w2) + self.softmax_b2

            # # crf
            self.tags_scores2 = tf.reshape(self.logits2, [self.batch_size, self.lstm_max_step, self.output_dim])
            self.log_likelihood2, self.transition_params2 = tf.contrib.crf.crf_log_likelihood(
                self.tags_scores2, self.output_task2, self.length, name='task2')
            self.batch_pred_sequence2, self.batch_viterbi_score2 = tf.contrib.crf.crf_decode(self.tags_scores2,
                                                                                             self.transition_params2,
                                                                                             self.length)

            self.loss_task2 = tf.reduce_mean(-self.log_likelihood2)

    def _build_task3(self):
        # attention_task3

        med_outputs3 = self.repre_LSTM

        with tf.name_scope("task3"):
            ### attention
            if self.is_attention:
                H = tf.reshape(med_outputs3, [-1, self.lstm_embedding_dims * 2])
                W_a = tf.get_variable("W_a_H3", shape=[self.lstm_embedding_dims * 2, self.attention_dim],
                                      initializer=self.initializer, trainable=True)
                H_u = tf.reshape(tf.matmul(H, W_a), [self.batch_size, self.lstm_max_step, self.lstm_embedding_dims * 2])

                position_H = tf.reshape(self.position_task3_wv, [-1, self.position_dim])
                W_a_p = tf.get_variable("W_a_pos3", shape=[self.position_dim, self.attention_dim],
                                        initializer=self.initializer, trainable=True)
                H_p = tf.reshape(tf.matmul(position_H, W_a_p),
                                 [self.batch_size, self.lstm_max_step, self.lstm_embedding_dims * 2])

                u = tf.matmul(H_u, H_p, transpose_b=True)
                self.alpha_task3 = tf.nn.softmax(u)
                context_3 = tf.matmul(self.alpha_task3, med_outputs3)
                self.attention_context_3 = tf.concat([context_3, self.attention_context_1], -1)
                self.attention_context_3 = tf.reshape(
                    tf.layers.dense(self.attention_context_3, self.lstm_embedding_dims * 2, name="linear3",
                                    use_bias=False),
                    [self.batch_size, self.lstm_max_step, self.lstm_embedding_dims * 2])
                med_outputs3 = self.attention_context_3

            # linear
            med_outputs3 = tf.reshape(med_outputs3, [-1, self.lstm_embedding_dims * 2])
            self.softmax_w3 = tf.get_variable("softmax_w3", [self.lstm_embedding_dims * 2, self.output_dim],
                                              initializer=self.initializer)
            self.softmax_b3 = tf.get_variable("softmax_b3", [self.output_dim], initializer=self.initializer)
            self.logits3 = tf.matmul(med_outputs3, self.softmax_w3) + self.softmax_b3

            # crf
            self.tags_scores3 = tf.reshape(self.logits3, [self.batch_size, self.lstm_max_step, self.output_dim])
            self.log_likelihood3, self.transition_params3 = tf.contrib.crf.crf_log_likelihood(
                self.tags_scores3, self.output_task3, self.length, name='task3')
            self.batch_pred_sequence3, self.batch_viterbi_score3 = tf.contrib.crf.crf_decode(self.tags_scores3,
                                                                                             self.transition_params3,
                                                                                             self.length)

            self.loss_task3 = tf.reduce_mean(-self.log_likelihood3)

    def train(self):
        X_train, y_train1, y_train2, y_train3, X_val, y_val1, y_val2, y_val3 = self.dataManager.getTrainingSet()
        tf.initialize_all_variables().run(session=self.sess)

        saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        tf.summary.merge_all()

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))
        num_val_iterations = int(math.ceil(1.0 * len(X_val) / self.batch_size))

        cnt = 0
        cnt_dev = 0
        unprogressed = 0
        very_start_time = time.time()
        self.logger.info("\ntraining starting" + ("+" * 20))
        for epoch in range(self.num_epochs):
            start_time = time.time()
            # shuffle train at each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train1 = y_train1[sh_index]
            y_train2 = y_train2[sh_index]
            y_train3 = y_train3[sh_index]

            self.logger.info("\ncurrent epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                X_train_batch, y_train_batch1, y_train_batch2, y_train_batch3 = self.dataManager.nextBatch(X_train,
                                                                                                           y_train1,
                                                                                                           y_train2,
                                                                                                           y_train3,
                                                                                                           start_index=iteration * self.batch_size)
                _, loss_train, train_batch_viterbi_sequence1, train_batch_viterbi_sequence2, train_batch_viterbi_sequence3 = \
                    self.sess.run([
                        self.opt_op,
                        self.loss,
                        self.batch_pred_sequence1,
                        self.batch_pred_sequence2,
                        self.batch_pred_sequence3,
                    ],
                        feed_dict={
                            self.text_input: X_train_batch,
                            self.output_task1: y_train_batch1,
                            self.output_task2: y_train_batch2,
                            self.output_task3: y_train_batch3,
                        })

                if iteration % self.print_per_batch == 0:
                    cnt += 1

                    measures = metrics(X_train_batch,
                                       y_train_batch1,
                                       y_train_batch2,
                                       y_train_batch3,
                                       train_batch_viterbi_sequence1,
                                       train_batch_viterbi_sequence2,
                                       train_batch_viterbi_sequence3,
                                       self.measuring_metrics, self.dataManager)

                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ": %.3f " % v)
                    self.logger.info("training batch: %5d, loss: %.5f, %s" % (iteration, loss_train, res_str))

            # validation
            loss_vals = list()
            val_results = dict()
            for measu in self.measuring_metrics:
                val_results[measu] = 0

            for iterr in range(num_val_iterations):
                cnt_dev += 1
                X_val_batch, y_val_batch1, y_val_batch2, y_val_batch3 \
                    = self.dataManager.nextBatch(X_val, y_val1, y_val2, y_val3,
                                                 start_index=iterr * self.batch_size)

                loss_val, val_batch_viterbi_sequence1, val_batch_viterbi_sequence2, val_batch_viterbi_sequence3 = \
                    self.sess.run([
                        self.loss,
                        self.batch_pred_sequence1,
                        self.batch_pred_sequence2,
                        self.batch_pred_sequence3,
                    ],
                        feed_dict={
                            self.text_input: X_val_batch,
                            self.output_task1: y_val_batch1,
                            self.output_task2: y_val_batch2,
                            self.output_task3: y_val_batch3,
                        })

                measures = metrics(X_val_batch, y_val_batch1, y_val_batch2, y_val_batch3,
                                   val_batch_viterbi_sequence1,
                                   val_batch_viterbi_sequence2,
                                   val_batch_viterbi_sequence3,
                                   self.measuring_metrics, self.dataManager)

                for k, v in measures.items():
                    val_results[k] += v
                loss_vals.append(loss_val)

            time_span = (time.time() - start_time) / 60
            val_res_str = ''
            dev_f1_avg = 0
            for k, v in val_results.items():
                val_results[k] /= num_val_iterations
                val_res_str += (k + ": %.3f " % val_results[k])
                if k == 'f1': dev_f1_avg = val_results[k]

            self.logger.info("time consumption:%.2f(min),  validation loss: %.5f, %s" %
                             (time_span, np.array(loss_vals).mean(), val_res_str))
            if np.array(dev_f1_avg).mean() > self.best_f1_val:
                unprogressed = 0
                self.best_f1_val = np.array(dev_f1_avg).mean()
                saver.save(self.sess, self.checkpoints_dir + "/" + self.checkpoint_name, global_step=self.global_step)
                self.logger.info("saved the new best model with f1: %.3f" % (self.best_f1_val))
            else:
                unprogressed += 1

            if self.is_early_stop:
                if unprogressed >= self.patient:
                    self.logger.info("early stopped, no progress obtained within %d epochs" % self.patient)
                    self.logger.info("final best f1 is: %f" % (self.best_f1_val))
                    self.sess.close()
                    return
        self.logger.info("final best f1 is: %f" % (self.best_f1_val))
        self.logger.info("total training time consumption: %.3f(min)" % ((time.time() - very_start_time) / 60))
        self.sess.close()

    def test(self):
        X_test, y_test_psyduo_label1, y_test_psyduo_label2, y_test_psyduo_label3, X_test_str = self.dataManager.getTestingSet()

        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        self.logger.info("total number of testing iterations: " + str(num_iterations))

        self.logger.info("loading model parameter\n")
        tf.initialize_all_variables().run(session=self.sess)
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoints_dir))

        tokens = []
        labels1 = []
        labels2 = []
        labels3 = []
        self.logger.info("\ntesting starting" + ("+" * 20))
        for i in range(num_iterations):
            self.logger.info("batch: " + str(i + 1))
            X_test_batch = X_test[i * self.batch_size: (i + 1) * self.batch_size]
            X_test_str_batch = X_test_str[i * self.batch_size: (i + 1) * self.batch_size]
            y_test_psyduo_label_batch1 = y_test_psyduo_label1[i * self.batch_size: (i + 1) * self.batch_size]
            y_test_psyduo_label_batch2 = y_test_psyduo_label2[i * self.batch_size: (i + 1) * self.batch_size]
            y_test_psyduo_label_batch3 = y_test_psyduo_label3[i * self.batch_size: (i + 1) * self.batch_size]

            if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                X_test_batch = list(X_test_batch)
                X_test_str_batch = list(X_test_str_batch)
                y_test_psyduo_label_batch1 = list(y_test_psyduo_label_batch1)
                y_test_psyduo_label_batch2 = list(y_test_psyduo_label_batch2)
                y_test_psyduo_label_batch3 = list(y_test_psyduo_label_batch3)
                gap = self.batch_size - len(X_test_batch)

                X_test_batch += [[0 for j in range(self.lstm_max_step)] for i in range(gap)]
                X_test_str_batch += [['x' for j in range(self.lstm_max_step)] for i in
                                     range(gap)]
                y_test_psyduo_label_batch1 += [[self.dataManager.label2id['O'] for j in range(self.lstm_max_step)] for i
                                               in range(gap)]
                y_test_psyduo_label_batch2 += [[self.dataManager.label2id['O'] for j in range(self.lstm_max_step)] for i
                                               in range(gap)]
                y_test_psyduo_label_batch3 += [[self.dataManager.label2id['O'] for j in range(self.lstm_max_step)] for i
                                               in range(gap)]
                X_test_batch = np.array(X_test_batch)
                X_test_str_batch = np.array(X_test_str_batch)
                y_test_psyduo_label_batch1 = np.array(y_test_psyduo_label_batch1)
                y_test_psyduo_label_batch2 = np.array(y_test_psyduo_label_batch2)
                y_test_psyduo_label_batch3 = np.array(y_test_psyduo_label_batch3)
                results1, results2, results3, token = self.predictBatch(self.sess, X_test_batch,
                                                                        y_test_psyduo_label_batch1,
                                                                        y_test_psyduo_label_batch2,
                                                                        y_test_psyduo_label_batch3,
                                                                        X_test_str_batch)
                results1 = results1[:len(X_test_batch)]
                results2 = results2[:len(X_test_batch)]
                results3 = results3[:len(X_test_batch)]
                token = token[:len(X_test_batch)]
            else:
                results1, results2, results3, token = self.predictBatch(self.sess, X_test_batch,
                                                                        y_test_psyduo_label_batch1,
                                                                        y_test_psyduo_label_batch2,
                                                                        y_test_psyduo_label_batch3,
                                                                        X_test_str_batch)

            labels1.extend(results1)
            labels2.extend(results2)
            labels3.extend(results3)
            tokens.extend(token)

        def save_test_out(tokens, labels1, labels2, labels3):
            # transform format
            newtokens, newlabels1, newlabels2, newlabels3 = [], [], [], []
            for to, la1, la2, la3 in zip(tokens, labels1, labels2, labels3):
                newtokens.extend(to)
                newtokens.append("")
                newlabels1.extend(la1)
                newlabels1.append("")
                newlabels2.extend(la2)
                newlabels2.append("")
                newlabels3.extend(la3)
                newlabels3.append("")
            # save
            save_csv_(pd.DataFrame({
                "token": newtokens, "label1": newlabels1, "label2": newlabels1, "label3": newlabels3
            }), self.output_test_file, ["token", "label1", "label2", "label3"],
                delimiter=self.configs.delimiter)

        save_test_out(tokens, labels1, labels2, labels3)
        self.logger.info("testing results saved.\n")

        self.sess.close()

    def predictBatch(self, sess, X, y_psydo_label1, y_psydo_label2, y_psydo_label3, X_test_str_batch):
        tokens = []
        predicts_labels_tokenlevel1 = []
        predicts_labels_tokenlevel2 = []
        predicts_labels_tokenlevel3 = []

        predicts_label_id1, predicts_label_id2, predicts_label_id3, lengths = \
            sess.run([
                self.batch_pred_sequence1,
                self.batch_pred_sequence2,
                self.batch_pred_sequence3,
                self.length
            ],
                feed_dict={
                    self.text_input: X,
                    self.output_task1: y_psydo_label1,
                    self.output_task2: y_psydo_label2,
                    self.output_task3: y_psydo_label3,
                })

        for i in range(len(lengths)):
            x_ = [val for val in X_test_str_batch[i, 0:lengths[i]]]
            tokens.append(x_)

            y_pred1 = [str(self.dataManager.id2label[val]) for val in predicts_label_id1[i, 0:lengths[i]]]
            predicts_labels_tokenlevel1.append(y_pred1)

            y_pred2 = [str(self.dataManager.id2label[val]) for val in predicts_label_id2[i, 0:lengths[i]]]
            predicts_labels_tokenlevel2.append(y_pred2)

            y_pred3 = [str(self.dataManager.id2label[val]) for val in predicts_label_id3[i, 0:lengths[i]]]
            predicts_labels_tokenlevel3.append(y_pred3)

        return predicts_labels_tokenlevel1, predicts_labels_tokenlevel2, predicts_labels_tokenlevel3, \
               tokens
