#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/12/22 21:49
@desc:
"""
from functools import reduce
from operator import mul
import tensorflow as tf
from tensorflow.keras import layers


FLOAT_TYPE = tf.float32


def tensordot(tensor_a, tensor_b):
    """ Tensor dot function. The last dimension of tensor_a and the first dimension of tensor_b must be the same.
    :param tensor_a:
    :param tensor_b:
    :return: the result of tensor_a tensor dot tensor_b.
    """
    last_idx_a = len(tensor_a.get_shape().as_list()) - 1
    return tf.tensordot(tensor_a, tensor_b, [[last_idx_a], [0]])


def swap_axes(tensor, axis1, axis2):
    """Interchange two axes of an tensor.
    :param tensor:
    :param axis1: First axis.
    :param axis2: Second axis.
    :return:
    """
    tensor_perm = list(range(len(tensor.shape.as_list())))
    tensor_perm[axis1] = axis2
    tensor_perm[axis2] = axis1

    return tf.transpose(tensor, perm=tensor_perm)


def create_tensor(shape, value):
    """Creates a tensor with all elements set to value and dtype is same sa value.
    :param shape: a list
    :param value: a number
    :return:
    """
    tensor_shape = tf.stack(shape)
    return tf.fill(tensor_shape, value)


def get_variable_weights(name, shape, collections=None):
    return tf.get_variable(name, shape=shape, dtype=FLOAT_TYPE,
                           initializer=tf.glorot_normal_initializer(),
                           collections=collections)


def get_variable_bias(name, shape, collections=None):
    return tf.get_variable(name, shape=shape, dtype=FLOAT_TYPE,
                           initializer=tf.constant_initializer(0.1),
                           collections=collections)


def get_num_trainable_params():
    """ Get the number of trainable parameters in current session (model).
    :return:
    """
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


class Attention:
    def __init__(self, cell_units, score_method='general'):
        self.cell_units = cell_units

        if score_method == 'general':
            attention_w1 = layers.Dense(self.cell_units, name='w1')
            attention_w2 = layers.Dense(self.cell_units, name='w2')
            attention_v = layers.Dense(1, name='v')
            score_fn = lambda q, k: attention_v(tf.nn.tanh(attention_w1(q) + attention_w2(k))) / tf.sqrt(
                tf.cast(self.cell_units, dtype=tf.float32))
        elif score_method == 'exp_dot':
            score_fn = lambda q, k: tf.reduce_sum(tf.exp(q * k), axis=-1, keepdims=True)
        else:
            raise RuntimeError('Unknown score_method:' + score_method)
        self.score_fn = score_fn

    def compute_attention_weight(self, queries, keys, values, pos_mask=None):
        """
        :param queries: (batch_size, n_queries, hidden_dim)
        :param keys: (batch_size, n_keys, hidden_dim)
        :param values: (batch_size, n_values, hidden_dim)
        :param pos_mask: ['self-right', 'right', None]
        self-right: mask values for the upper right area, excluding the diagonal
        right: mask values for the upper right area, including the diagonal
        None: no mask.
        :return: (batch_size, num_queries, cell_units), (batch_size, num_queries, num_keys, 1)
        """
        MASKED_VAL = - 2 ** 32 + 1
        # (batch_size, num_queries, 1, hidden_dim)
        q = tf.expand_dims(queries, axis=2)
        # (batch_size, 1, num_keys, hidden_dim)
        k = tf.expand_dims(keys, axis=1)
        v = tf.expand_dims(values, axis=1)

        # (batch_size, num_queries, num_keys, 1)
        score = self.score_fn(q, k)

        if pos_mask:
            # (batch_size, num_queries, num_keys)
            score = tf.squeeze(score, axis=-1)

            ones_mat = tf.ones_like(score)
            zeros_mat = tf.zeros_like(score)
            masked_val_mat = ones_mat * MASKED_VAL

            # (batch_size, num_queries, num_keys)
            lower_diag_masks = tf.linalg.LinearOperatorLowerTriangular(ones_mat).to_dense()

            if pos_mask == 'right':
                # mask values for the upper right area, including the diagonal
                # (batch_size, num_queries, num_keys)
                score = tf.where(tf.equal(lower_diag_masks, 0),
                                 masked_val_mat,
                                 score)
                attention_weight = tf.nn.softmax(score, axis=-1)
                attention_weight = tf.where(tf.equal(lower_diag_masks, 0),
                                            zeros_mat,
                                            attention_weight)
            elif pos_mask == 'self-right':
                # mask values for the upper right area, excluding the diagonal
                # transpose to upper triangle
                lower_masks = tf.transpose(lower_diag_masks, perm=[0, 2, 1])

                score = tf.where(tf.equal(lower_masks, 1),
                                 masked_val_mat,
                                 score)
                attention_weight = tf.nn.softmax(score, axis=-1)
                attention_weight = tf.where(tf.equal(lower_masks, 1),
                                            zeros_mat,
                                            attention_weight)

            else:
                raise RuntimeError('Unknown pas_mask: {}'.format(pos_mask))

            # (batch_size, num_queries, num_keys, 1)
            attention_weight = tf.expand_dims(attention_weight, axis=-1)
        else:
            # (batch_size, num_queries, num_keys, 1)
            attention_weight = tf.nn.softmax(score, axis=2)

        # (batch_size, num_queries, num_keys, cell_units)
        context_vector = attention_weight * v

        # (batch_size, num_queries, cell_units)
        context_vector = tf.reduce_sum(context_vector, axis=2)

        return context_vector, attention_weight
