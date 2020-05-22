#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: Transformer
@time: 2020/5/22 17:29
@desc:
"""
import tensorflow as tf
from tensorflow.keras import layers

from models.base_model import BaseModel
from models.modules import multi_head_attention, layer_norm, ffn, label_smoothing, positional_encoding_map


class TransformerRegression(BaseModel):
    """
    Transformer for regression.
    """

    def train(self, sess, batch_data, **kwargs):
        # get parameter
        lr = kwargs.get('lr')
        dropout_rate = kwargs.get('dropout_rate', 0.0)

        fd = {
            self.x_ph: batch_data[0],
            self.y_ph: batch_data[1],
            self.lr_ph: lr,
            self.dropout_ph: dropout_rate
        }
        _, loss, pred, real = sess.run([self.train_op, self.loss, self.y_hat, self.y_ph], feed_dict=fd)
        return loss, pred[:, :-1], real[:, 1:]

    def predict(self, sess, batch_data, **kwargs):
        fd = {
            self.x_ph: batch_data[0],
            self.y_ph: batch_data[1],
            self.dropout_ph: 0.0
        }
        loss, pred, real = sess.run([self.loss, self.y_hat, self.y_ph], feed_dict=fd)
        return loss, pred[:, :-1], real[:, 1:]

    def __init__(self, model_config, padding_flag=0):
        # super class initialization
        super(BaseModel, self).__init__()

        self.padding_flag = padding_flag
        # get the parameters of model config
        self.N = model_config.get('N')
        self.n_heads = model_config.get('n_heads')
        self.hidden_dim = model_config.get('hidden_dim')
        self.ffn_dim = model_config.get('ffn_dim')

        self.D = model_config.get('D')
        self.T = model_config.get('T')
        self.h = model_config.get('h')
        # model input
        with tf.variable_scope('model_input'):
            self.x_ph = tf.placeholder(tf.float32, shape=[None, self.T, self.D])
            self.y_ph = tf.placeholder(tf.float32, shape=[None, self.h, self.D])

            # training placeholder
            self.lr_ph = tf.placeholder(tf.float32)
            self.dropout_ph = tf.placeholder(tf.float32)

        with tf.variable_scope('transformer'):
            # process mask
            # shape -> [batch_size, seq_len]
            src_mask = None
            tgt_mask = None

            # --- Inputs and Outputs Embedding ---
            self.emb_weights = tf.get_variable('emb_weights', shape=[self.D, self.hidden_dim],
                                               initializer=tf.glorot_uniform_initializer())
            emb_bias = tf.get_variable('emb_bias', shape=[self.hidden_dim],
                                       initializer=tf.constant_initializer(0.1))
            # shape -> [batch_size, seq_len, hidden_dim]
            src_embeddings = tf.einsum('btd,dk->btk', self.x_ph, self.emb_weights) + emb_bias
            tgt_embeddings = tf.einsum('btd,dk->btk', self.y_ph, self.emb_weights) + emb_bias

            # # --- Positional Encoding ---
            # # get positional encoding of source and target sequence
            # # shape -> [max_len, hidden_dim]
            # pe_map = tf.convert_to_tensor(positional_encoding_map(self.max_len, self.hidden_dim),
            #                               dtype=tf.float32)
            #
            # src_seq_len = tf.shape(self.x_ph)[1]
            # tgt_seq_len = tf.shape(self.y_ph)[1]
            # # shape -> [1, seq_len, hidden_dim]
            # src_pe = pe_map[None, :src_seq_len]
            # tgt_pe = pe_map[None, :tgt_seq_len]
            #
            # # add positional encoding to embedding element-wisely
            # src_embeddings += src_pe
            # tgt_embeddings += tgt_pe
            #
            # # use dropout
            # src_embeddings = self.use_dropout(src_embeddings)
            # tgt_embeddings = self.use_dropout(src_embeddings)

            # --- Encoder ---
            enc_output = self.encode(src_embeddings, src_mask)
            # --- Decoder ---
            dec_output = self.decode(tgt_embeddings, enc_output, tgt_mask, src_mask)
            # --- Inference ---
            # shape -> [batch_size, seq_len], [batch_size, seq_len, process_dim]
            y_hat = self.inference(dec_output)

            # Loss Function
            loss = tf.reduce_mean(tf.abs(y_hat[:, :-1] - self.y_ph[:, 1:]))

            optimizer = tf.train.AdamOptimizer(self.lr_ph)
            train_op = optimizer.minimize(loss)

            # assigning
            self.train_op = train_op
            self.loss = loss
            self.y_hat = y_hat

    def gen_embedding_map(self, process_dim, hidden_dim, name):
        """ Generate embedding map.
        :param process_dim: the depth of one hot seq
        :param hidden_dim: the dimension of embeedding
        :param name: name for embedding variable
        :return: the embeddings with shape [process_dim, hidden_dim]
        """
        embeddings = tf.get_variable(name, shape=[process_dim, hidden_dim], dtype=tf.float32,
                                     initializer=tf.glorot_uniform_initializer())
        return embeddings * hidden_dim ** 0.5

    def encode(self, input_seq, seq_mask):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            enc_output = input_seq
            for i in range(self.N):
                with tf.variable_scope('layer_{}'.format(i + 1), reuse=tf.AUTO_REUSE):
                    # multi-head attention
                    mh_output = multi_head_attention(queries=enc_output,
                                                     keys=enc_output,
                                                     values=enc_output,
                                                     n_heads=self.n_heads,
                                                     key_mask=seq_mask,
                                                     causality=False,
                                                     scope='mh_att')

                    # use dropout
                    mh_output = self.use_dropout(mh_output)
                    # short-cut and layer normalization
                    mh_output = layer_norm(enc_output + mh_output)

                    # feed forward network
                    ff_output = ffn(mh_output,
                                    dims=[self.ffn_dim, self.hidden_dim],
                                    scope='ffn')

                    # use dropout
                    ff_output = self.use_dropout(ff_output)
                    # short-cut and layer normalization
                    ff_output = layer_norm(mh_output + ff_output)

                    enc_output = ff_output

            return enc_output

    def decode(self, target_seq, source_seq, target_mask, source_mask):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            dec_output = target_seq
            for i in range(self.N):
                with tf.variable_scope('layer_{}'.format(i + 1), reuse=tf.AUTO_REUSE):
                    # masked multi-head attention for target_seq
                    mask_mh_output = multi_head_attention(queries=dec_output,
                                                          keys=dec_output,
                                                          values=dec_output,
                                                          n_heads=self.n_heads,
                                                          key_mask=target_mask,
                                                          causality=True,
                                                          scope='masked_mh_att')

                    # use dropout
                    mask_mh_output = self.use_dropout(mask_mh_output)
                    # short-cut and layer normalization
                    mask_mh_output = layer_norm(dec_output + mask_mh_output)

                    # multi-head attention for enc_output and target_seq
                    mh_output = multi_head_attention(queries=mask_mh_output,
                                                     keys=source_seq,
                                                     values=source_seq,
                                                     n_heads=self.n_heads,
                                                     key_mask=source_mask,
                                                     causality=False,
                                                     scope='mh_att_with_enc_input')
                    # use dropout
                    mh_output = self.use_dropout(mh_output)
                    # short-cut and layer normalization
                    mh_output = layer_norm(mask_mh_output + mh_output)

                    # feed forward network
                    ff_output = ffn(mh_output,
                                    dims=[self.ffn_dim, self.hidden_dim],
                                    scope='ffn')

                    # use dropout
                    ff_output = self.use_dropout(ff_output)
                    # short-cut and layer normalization
                    ff_output = layer_norm(mh_output + ff_output)

                    dec_output = ff_output

        return dec_output

    def inference(self, dec_output):
        """ Inference the final result.
        :param dec_output: a tensor with shape [batch_size, seq_len, hidden_dim]
        :return:
        y_hat: a tensor with shape [batch_size, seq_len], which contains the index of predicted point.
        """
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            # using liner projection to obtain final result
            # shape -> [batch_size, seq_len, process_dim]
            bias = tf.get_variable('bias', shape=[self.D],
                                   initializer=tf.constant_initializer(0.1))
            y_hat = tf.einsum('btk,dk->btd', dec_output, self.emb_weights) + bias

        return y_hat

    def use_dropout(self, x):
        return tf.nn.dropout(x, rate=self.dropout_ph)
