#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/6 16:42
@desc:
"""
import pickle
from preprocess.data_source import DataSource


def get_static_data_callback(data):
    """
    callback generator for getting data online
    :param data:
    :return:
    """

    def data_callback():
        yield data

    return data_callback


class DataLoader(object):

    def __init__(self, data_config):
        self._data_name = data_config['data_name']
        self._data_filename = data_config['data_filename']
        self._cache_dir = data_config['cache_dir']
        self._use_cache = data_config['use_cache']
        self._process_dim = data_config['process_dim']

    def get_three_datasource(self):
        """ Load the raw data, and then return three data sources containing train data, validation and test
        data separately.
        :return: train, validation and test DataSource.
        """
        # load data
        if self._use_cache:
            # read data from cache
            train_ds = DataSource(self._data_name + '_train', cache_dir=self._cache_dir)
            valid_ds = DataSource(self._data_name + '_valid', cache_dir=self._cache_dir)
            test_ds = DataSource(self._data_name + '_test', cache_dir=self._cache_dir)
        else:
            # keys(types, timesteps)   format:n_seqs * [seq_len]
            with open(self._data_filename.format('train'), 'rb') as f:
                train_records = pickle.load(f)

            with open(self._data_filename.format('valid'), 'rb') as f:
                valid_records = pickle.load(f)

            with open(self._data_filename.format('test'), 'rb') as f:
                test_records = pickle.load(f)

            # wrapping data into DataSource
            train_ds = DataSource(self._data_name + '_train', cache_dir=self._cache_dir,
                                  retrieve_data_callback=get_static_data_callback(train_records))
            valid_ds = DataSource(self._data_name + '_valid', cache_dir=self._cache_dir,
                                  retrieve_data_callback=get_static_data_callback(valid_records))
            test_ds = DataSource(self._data_name + '_test', cache_dir=self._cache_dir,
                                 retrieve_data_callback=get_static_data_callback(test_records))

        return train_ds, valid_ds, test_ds
