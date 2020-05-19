#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: Transformer
@time: 2020/5/19 18:59
@desc:
"""
import tensorflow as tf
from tensorflow.keras import layers


def layer_norm(inputs, epsilon=1e-8, scope='layer_norm'):
    """
    :param inputs: a tensor.
    :param epsilon: a float number for preventing zero division.
    :param scope:
    :return: a tensor with the same shape and data type as 'inputs'.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        param_shape = inputs.get_shape()[-1:]
        gamma = tf.get_variable('gamma', param_shape, initializer=tf.ones_initializer())
        beta = tf.get_variable('beta', param_shape, initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        inputs_norm = (inputs - mean) / (variance + epsilon ** 0.5)
        output = gamma * inputs_norm + beta

        return output


def multi_head_attention(queries, keys, values, n_heads, key_mask, causality, scope):
    """ Split the input into n_heads heads, then calculate the context vector for each head, and merge all
    context vectors into output.
    :param queries: the query sequences. [..., n_queries, hidden_dim]
    :param keys: the key sequences. [..., n_keys, hidden_dim]
    :param values: the value sequences whose length is same as keys. [..., n_keys, hidden_dim]
    :param n_heads: the number of heads
    :param key_mask: mask for keys. [..., n_keys]
    :param causality: mask for queries. True or False
    :param scope: the variable scope name
    :return: context vector. [..., n_queries, hidden_dim]
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

        # [n_heads, ..., n_queries, hidden_dim / n_heads]p
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


def ffn(x, dims, activation=tf.nn.relu, scope='ffn'):
    """ Feed Forward Network.
    :param x:
    :param dims: a list of each layer dimension.
    :param activation: activation function for inner layer.
    :param scope:
    :return:
    """

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for dim in dims[:-1]:
            x = layers.Dense(dim, activation=activation)(x)

        output = layers.Dense(dims[-1])(x)

        return output
