#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/5 21:01
@desc:
"""
from abc import abstractmethod

from lib.utils import yield2batch_data, get_metrics_callback_from_names
from lib.scalers import DictScaler, VoidScaler, ZeroMaxScaler, SingletonStandScaler


class AbstractDataProvider(object):

    def __init__(self, data_source, data_config):
        self._data_source = data_source
        self._batch_size = data_config['batch_size']
        self._metrics_function = get_metrics_callback_from_names(data_config['metrics'])

    def get_metrics(self, preds, labels):
        """ Calculate the metrics of preds and labels.
        :param preds:
        :param labels:
        :return: a dictionary of the metrics.
        """
        return self._metrics_function(preds, labels)

    @abstractmethod
    def iterate_batch_data(self):
        """ Get batch model input of one epoch.
        Remark: batch -> partition -> epoch
        :return: yield a list containing batch inputs until the end of the epoch.
        """
        pass

    @abstractmethod
    def epoch_inverse_scaling(self, scaled_records):
        """ Inverse the scaled_records to real scale.
        :param scaled_records:
        :return: real scale records.
        """
        pass


# ----------------- Template DataProvider ---------------------
class DataProvider(AbstractDataProvider):
    """
    Data provider for processing model inputs.
    """
    def __init__(self, data_source, data_config, scaler=None):
        super(DataProvider, self).__init__(data_source, data_config)
        self._scaler = scaler if scaler else DictScaler(dtimes=VoidScaler, marks=SingletonStandScaler)
        self._is_first_iterate = True

        self._type_padding = data_config['process_dim']

    def epoch_inverse_scaling(self, scaled_records):
        return self._scaler.inverse_scaling(scaled_records)

    def iterate_batch_data(self):
        """ Iterate one batch input for model over whole dataset.
        :return:
        """
        # record_data of a partition whose shape is [n_records, ...]
        for data in self._data_source.load_partition_data():
            if self._is_first_iterate:
                data_stats = self._dataset_statistics(data)
                print(f'Load dataset {self._data_source.data_name}: {data_stats}')

            inputs = self._process_model_input(data)
            if self._scaler.is_fit():
                scaled_inputs = self._scaler.scaling(inputs)
            else:
                scaled_inputs = self._scaler.fit_scaling(inputs)

            # yield records to batch data separately
            for batch_data in yield2batch_data(scaled_inputs, self._batch_size, keep_remainder=True):
                yield batch_data

        if self._is_first_iterate:
            self._is_first_iterate = False

    def get_scaler(self):
        return self._scaler

    def _process_model_input(self, data):
        """ Process each item as model input.
        :param records:
        :return:
        """
        ret = None
        return ret

    def _dataset_statistics(self, data):
        """ Describe the data.
        :param data:
        :return: The description of data.
        """
        data_stats = None
        return data_stats
