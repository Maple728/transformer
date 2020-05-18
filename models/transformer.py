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
        self.config.input_process_dim = model_config.get('input_process_dim')
        self.config.output_process_dim = model_config.get('output_process_dim')

        # model input
        with tf.variable_scope('model_input'):
            # shape -> [batch_size, seq_len]
            self.x_ph = tf.placeholder(tf.int32, shape=[None, None])
            self.y_ph = tf.placeholder(tf.int32, shape=[None, None])

            # training placeholder
            self.lr_ph = tf.placeholder(tf.float32)

    def embedding(self, seq, process_dim, hidden_dim):
        """Embed each discrete point in seq.
        :param seq: with shape [...]
        :param process_dim: the depth of one hot seq
        :param hidden_dim: the dimension of embeedding
        :return: the embedding of seq with shape [..., hidden_dim]
        """
        # onehot seq, shape -> [..., process_dim]
        seq_onehot = tf.one_hot(seq, process_dim)
        # embed onehot point using simple embedding, shape -> [..., hidden_dim]
        seq_embedding = layers.Embedding(process_dim, hidden_dim)(seq_onehot)

        return seq_embedding

    def positional_encoding(self, seq):
        return None


def multi_head_attention(queries, keys, values, n_heads, key_mask, causality, scope):
    """

    :param queries: the query sequences. [..., n_queries, hidden_dim]
    :param keys: the key sequences. [..., n_keys, hidden_dim]
    :param values: the value sequences whose length is same as keys. [..., n_keys, hidden_dim]
    :param n_heads: the number of heads
    :param key_mask: mask for keys. [..., n_keys]
    :param causality: mask for queries. True or False
    :param scope: the variable scope name
    :return:
    """

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        hidden_dim = queries.get_shape().as_list()[-1]
        # transform input
        queries = layers.Dense(hidden_dim, name='Q_dense')(queries)
        keys = layers.Dense(hidden_dim, name='K_dense')(keys)
        values = layers.Dense(hidden_dim, name='V_dense')(values)

        # split the whole input into the part input for each head
        # [n_heads, ..., n_queries, hidden_dim / n_heads]
        queries = tf.stack(tf.split(queries, n_heads, axis=-1), axis=0)
        # [n_heads, ..., n_keys, hidden_dim / n_heads]
        keys = tf.stack(tf.split(keys, n_heads, axis=-1), axis=0)
        # [n_heads, ..., n_keys, hidden_dim / n_heads]
        values = tf.stack(tf.split(values, n_heads, axis=-1), axis=0)

        # [n_heads, ..., n_queries, hidden_dim / n_heads]
        context_vector = scaled_dot_product_attention(queries, keys, values, key_mask, causality)
        # [..., n_queries, hidden_dim]
        context_vector = tf.concat(tf.unstack(context_vector, axis=0), axis=-1)

        # merge all outputs of each head
        output = layers.Dense(hidden_dim, name='head_merge')(context_vector)

        return output


def scaled_dot_product_attention(queries, keys, values, key_mask=None, causality=False):
    """ Calculate the context vector using scaled dot product attention mechanism.

    :param queries: [..., n_queries, hidden_dim]
    :param keys: [..., n_keys, hidden_dim]
    :param values: [..., n_keys, hidden_dim]
    :param key_mask: mask for keys. [..., n_keys]
    :param causality: mask for queries. True or False.
    :return: context vector. [..., n_queries, hidden_dim]
    """

    with tf.name_scope('scaled_attention'):
        # general setting
        MASKED_VAL = - 2 ** 31
        score_fn = lambda q, k: tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(
            tf.cast(q.get_shape().as_list()[-1], dtype=tf.float32))

        score = score_fn(queries, keys)     # [..., n_queries, n_keys]

        # mask score by mask of keys
        if key_mask:
            key_mask_mat = key_mask * MASKED_VAL  # [..., n_keys]
            key_mask_mat = tf.expand_dims(key_mask_mat, -2)     # [..., 1, n_keys]
            score += key_mask_mat

        # mask score by causality of queries
        # mask values for the upper right area, including the diagonal
        if causality:

            ones_mat = tf.ones_like(score)
            zeros_mat = tf.zeros_like(score)
            masked_val_mat = ones_mat * MASKED_VAL

            # [..., n_queries, n_keys]
            lower_diag_masks = tf.linalg.LinearOperatorLowerTriangular(ones_mat).to_dense()

            score = tf.where(tf.equal(lower_diag_masks, 0),
                             masked_val_mat,
                             score)
            # [..., n_queries, n_keys]
            attention_weight = tf.nn.softmax(score, axis=-1)
            # attention_weight = tf.where(tf.equal(lower_diag_masks, 0),
            #                             zeros_mat,
            #                             attention_weight)

        else:
            # [..., n_queries, n_keys]
            attention_weight = tf.nn.softmax(score, axis=2)

        # [..., n_queries, hidden_dim]
        context_vector = tf.matmul(attention_weight, values)

        return context_vector
