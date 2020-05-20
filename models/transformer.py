#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: Transformer
@time: 2020/5/18 17:29
@desc:
"""
import tensorflow as tf
from tensorflow.keras import layers

from models.base_model import BaseModel
from models.modules import multi_head_attention, layer_norm, ffn, label_smoothing, positional_encoding_map


class Transformer(BaseModel):

    def train(self, sess, batch_data, **kwargs):
        pass

    def predict(self, sess, batch_data, **kwargs):
        pass

    def __init__(self, model_config, padding_flag=0):
        # super class initialization
        super(BaseModel, self).__init__()

        self.padding_flag = padding_flag
        # get the parameters of model config
        self.N = model_config.get('N')
        self.n_heads = model_config.get('n_heads')
        self.hidden_dim = model_config.get('hidden_dim')
        self.ffn_dim = model_config.get('ffn_dim')
        self.input_process_dim = model_config.get('input_process_dim')
        self.output_process_dim = model_config.get('output_process_dim')
        self.max_len = model_config.get('max_len', 256)

        # model input
        with tf.variable_scope('model_input'):
            # shape -> [batch_size, seq_len]
            self.x_ph = tf.placeholder(tf.int32, shape=[None, None])
            self.y_ph = tf.placeholder(tf.int32, shape=[None, None])

            # training placeholder
            self.lr_ph = tf.placeholder(tf.float32)
            self.dropout_ph = tf.placeholder(tf.float32)

        with tf.variable_scope('transformer'):
            # generate source and target sequence embedding map
            # shape -> [process_dim, hidden_dim]
            self.src_embedding_map = self.gen_embedding_map(self.input_process_dim,
                                                            self.hidden_dim,
                                                            'src_embedding_map')
            self.tgt_embedding_map = self.gen_embedding_map(self.output_process_dim,
                                                            self.hidden_dim,
                                                            'tgt_embedding_map')

            # process mask
            # shape -> [batch_size, seq_len]
            src_mask = tf.cast(tf.not_equal(self.x_ph, self.padding_flag), tf.float32)
            tgt_mask = tf.cast(tf.not_equal(self.y_ph, self.padding_flag), tf.float32)

            # --- Inputs and Outputs Embedding ---
            # embed source and target sequence using corresponding embedding map separately
            # shape -> [batch_size, seq_len, hidden_dim]
            src_embeddings = tf.nn.embedding_lookup(self.src_embedding_map, self.x_ph)
            tgt_embeddings = tf.nn.embedding_lookup(self.tgt_embedding_map, self.y_ph)

            # --- Positional Encoding ---
            # get positional encoding of source and target sequence
            # shape -> [max_len, hidden_dim]
            pe_map = tf.convert_to_tensor(positional_encoding_map(self.max_len, self.hidden_dim),
                                          dtype=tf.float32)

            src_seq_len = tf.shape(self.x_ph)[1]
            tgt_seq_len = tf.shape(self.y_ph)[1]
            # shape -> [1, seq_len, hidden_dim]
            src_pe = pe_map[None, :src_seq_len]
            tgt_pe = pe_map[None, :tgt_seq_len]

            # add positional encoding to embedding element-wisely
            src_embeddings += src_pe
            tgt_embeddings += tgt_pe

            # use dropout
            src_embeddings = self.use_dropout(src_embeddings)
            tgt_embeddings = self.use_dropout(src_embeddings)

            # --- Encoder ---
            enc_output = self.encode(src_embeddings, src_mask)
            # --- Decoder ---
            dec_output = self.decode(tgt_embeddings, enc_output, tgt_mask, src_mask)
            # --- Inference ---
            # shape -> [batch_size, seq_len], [batch_size, seq_len, process_dim]
            y_hat, logits = self.inference(dec_output)

            # Loss Function
            # shape -> [batch_size, seq_len, process_dim]
            y_ = label_smoothing(tf.one_hot(self.y_ph, depth=self.output_process_dim))
            # shape -> [batch_size, seq_len, process_dim]
            ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
            loss = tf.reduce_sum(ce * tf.expand_dims(tgt_mask, axis=-1)) / (tf.reduce_sum(tgt_mask) + 1e-8)

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
                                     initializer=tf.truncated_normal_initializer())
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
        logits: a tensor with shape [batch_size, seq_len, process_dim], which contains the logit for each point.
        """
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            # using liner projection to obtain final result
            # shape -> [batch_size, seq_len, process_dim]
            logits = tf.einsum('btd,kd->btk', dec_output, self.tgt_embedding_map)

            # shape -> [batch_size, seq_len]
            y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return y_hat, logits

    def use_dropout(self, x):
        return tf.nn.dropout(x, rate=self.dropout_ph)
