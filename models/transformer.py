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
from models.modules import multi_head_attention, layer_norm, ffn


class Transformer(BaseModel):

    def train(self, sess, batch_data, **kwargs):
        pass

    def predict(self, sess, batch_data, **kwargs):
        pass

    def __init__(self, model_config):
        # super class initialization
        super(BaseModel, self).__init__()

        # get the parameters of model config
        self.config = {}
        self.config.N = model_config.get('N')
        self.config.n_heads = model_config.get('n_heads')
        self.config.hidden_dim = model_config.get('hidden_dim')
        self.config.ffn_dim = model_config.get('ffn_dim')
        self.config.input_process_dim = model_config.get('input_process_dim')
        self.config.output_process_dim = model_config.get('output_process_dim')

        # model input
        with tf.variable_scope('model_input'):
            # shape -> [batch_size, seq_len]
            self.x_ph = tf.placeholder(tf.int32, shape=[None, None])
            self.y_ph = tf.placeholder(tf.int32, shape=[None, None])

            # training placeholder
            self.lr_ph = tf.placeholder(tf.float32)

    def get_embeddings(self, process_dim, hidden_dim, name):
        """Embed each discrete point in seq.
        :param process_dim: the depth of one hot seq
        :param hidden_dim: the dimension of embeedding
        :param name: name for embedding variable
        :return: the embeddings with shape [process_dim, hidden_dim]
        """
        embeddings = tf.get_variable(name, shape=[process_dim, hidden_dim], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer())
        return embeddings

    def positional_encoding(self, seq):
        return None

    def encode(self, input_seq, seq_mask):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            # process key_mask
            input_mask = None

            enc_output = input_seq
            for i in range(self.config.N):
                with tf.variable_scope('layer_{}'.format(i+1), reuse=tf.AUTO_REUSE):
                    # multi-head attention
                    mh_output = multi_head_attention(queries=enc_output,
                                                     keys=enc_output,
                                                     values=enc_output,
                                                     n_heads=self.config.n_heads,
                                                     key_mask=input_mask,
                                                     causality=False,
                                                     scope='mh_att')

                    # short-cut and layer normalization
                    mh_output = layer_norm(enc_output + mh_output)

                    # feed forward network
                    ff_output = ffn(mh_output,
                                    dims=[self.config.ffn_dim, self.config.hidden_dim],
                                    scope='ffn')

                    # short-cut and layer normalization
                    ff_output = layer_norm(mh_output + ff_output)

                    enc_output = ff_output

            return enc_output

    def decode(self, target_seq, source_seq, target_mask, source_mask):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):

            dec_output = target_seq
            for i in range(self.config.N):
                with tf.variable_scope('layer_{}'.format(i + 1), reuse=tf.AUTO_REUSE):
                    # masked multi-head attention for target_seq
                    mask_mh_output = multi_head_attention(queries=dec_output,
                                                          keys=dec_output,
                                                          values=dec_output,
                                                          n_heads=self.config.n_heads,
                                                          key_mask=target_mask,
                                                          causality=True,
                                                          scope='masked_mh_att')

                    # short-cut and layer normalization
                    mask_mh_output = layer_norm(dec_output + mask_mh_output)

                    # multi-head attention for enc_output and target_seq
                    mh_output = multi_head_attention(queries=mask_mh_output,
                                                     keys=source_seq,
                                                     values=source_seq,
                                                     n_heads=self.config.n_heads,
                                                     key_mask=source_mask,
                                                     causality=False,
                                                     scope='mh_att_with_enc_input')
                    # short-cut and layer normalization
                    mh_output = layer_norm(mask_mh_output + mh_output)

                    # feed forward network
                    ff_output = ffn(mh_output,
                                    dims=[self.config.ffn_dim, self.config.hidden_dim],
                                    scope='ffn')

                    # short-cut and layer normalization
                    ff_output = layer_norm(mh_output + ff_output)

                    dec_output = ff_output

        return dec_output

    def inference(self, x):
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            tf.einsum
            x = layers.Dense(1, )