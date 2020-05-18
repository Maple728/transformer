#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/6 14:58
@desc:
"""
import os
import pickle

from lib.utils import create_folder


class DataSource(object):
    """
    Data source for loading and caching data, and owns data scaler and corresponding metric function.
    """
    def __init__(self, data_name,
                 cache_dir,
                 retrieve_data_callback=None):
        """
        :param data_name:
        :param cache_dir:
        :param retrieve_data_callback: None means the data is cached, otherwise the data is not cached.
        """
        self._data_name = data_name
        self._retrieve_data_callback = retrieve_data_callback

        # process cache path and flag
        self.cache_path = create_folder(cache_dir, self._data_name)
        self.is_cached = False if self._retrieve_data_callback else True

    @property
    def data_name(self):
        return self._data_name

    def load_partition_data(self):
        """Iterate data from callback function or disk cache. The data is an array containing records, whose first dimension
        is the number of records.
        :return: [feat_arr, target_arr]
        """
        if self.is_cached:
            # load all partitions data from cache in disk
            partition_count = 0
            for filename in os.listdir(self.cache_path):
                filepath = os.path.join(self.cache_path, filename)

                yield self._load_records(filepath)

                partition_count += 1

            # check whether the data is cached.
            if partition_count == 0:
                raise RuntimeError("The data isn't cached")
        else:
            # load all partitions data from data callback online.
            for i, records in enumerate(self._retrieve_data_callback()):
                # cache data into disk
                filepath = os.path.join(self.cache_path, str(i) + '.pkl')
                self._save_records(filepath, records)

                yield records
            # set cached flag to true
            self.is_cached = True

    @staticmethod
    def _save_records(filepath, records):
        with open(filepath, 'wb') as file:
            pickle.dump(records, file, protocol=2)

    @staticmethod
    def _load_records(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
