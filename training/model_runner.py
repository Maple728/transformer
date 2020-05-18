#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: EmptyModel
@time: 2019/11/5 15:48
@desc:
"""

import os
import datetime

import numpy as np
import tensorflow as tf
import yaml

from lib.utils import make_config_string, create_folder, concat_arrs_of_dict, Timer, get_logger
from lib.tf_utils import get_num_trainable_params
from training.lr_scheduler import LRScheduler
from models import *


class ModelRunner(object):
    """
    Train, evaluate, save and restore model.
    """

    def __init__(self, config):
        self._config = config
        self._train_config = config['train']
        self._model_config = config['model']
        self._data_config = config['data']

        # the folders for model and tensor board
        self._training_folder = None
        self._model_folder = None
        self._tfb_folder = None

        # build model
        self._model = BaseModel.generate_model_from_config(self._model_config)
        self._model_saver = tf.train.Saver(max_to_keep=0)

        # other setting
        self._timer = Timer('m')

    @property
    def model(self):
        return self._model

    def train_model(self, sess,
                    train_data_provider, valid_data_provider, test_data_provider=None):

        epoch_num, max_epoch, lr_scheduler, continue_training = self._load_train_status()

        # get logger for this training
        logger = get_logger(os.path.join(self._training_folder, 'training.log'))

        # training from scratch or continue training
        if continue_training:
            # trained model existed, then restore it.
            model_path = self.restore_model(sess)
            epoch_num += 1
            logger.info(f'Restore model from {model_path}')
        else:
            # initialize variables
            sess.run([tf.global_variables_initializer()])

        logger.info(f'Training starts on dataset {self._data_config["data_name"]}')
        logger.info(f'----------Trainable parameter count: {get_num_trainable_params()} of model {self._model_folder}')

        best_valid_loss = float('inf')
        lr = lr_scheduler.get_lr()
        while lr > 0 and epoch_num <= max_epoch:

            # Train
            loss, _, _, elapse = self._run_epoch(sess, train_data_provider, lr, is_train=True)

            logger.info(f'Epoch {epoch_num}: train loss - {loss}, learning rate - {lr}.'
                        f' Cost time: {elapse:.3f}{self._timer.unit()}')

            # Valid
            loss, _, _, _ = self._run_epoch(sess, valid_data_provider, lr, is_train=False)

            # Update after train and valid
            # update lr
            lr = lr_scheduler.update_lr(loss=loss, epoch_num=epoch_num)
            # update train_config
            self._update_train_config(lr, epoch_num)

            if loss < best_valid_loss:
                best_valid_loss = loss

                # save best model
                self._save_model_with_config(sess)

                # Test
                loss, preds, labels, elapse = self._run_epoch(sess, test_data_provider, lr, is_train=False)
                metrics = test_data_provider.get_metrics(preds, labels)
                str_metrics = str(metrics)
                logger.info(f'---Test Loss: {loss}, metrics: {str_metrics}. '
                            f'Cost time: {elapse:.3f}{self._timer.unit()}')

            epoch_num += 1
        logger.info('Training Finished!')

    def evaluate_model(self, sess, data_provider):
        self.restore_model(sess)
        loss, preds, labels, _ = self._run_epoch(sess, data_provider,
                                                 lr=0, is_train=False)
        metrics = data_provider.get_metrics(preds, labels)
        return preds, labels, metrics

    def restore_model(self, sess):
        train_config = self._train_config
        model_path = train_config['model_path']
        self._model_saver.restore(sess, model_path)
        return model_path

    def _load_train_status(self):
        """ Load training status. Create base folders if the config presents a new training.
        :return:
        """
        train_config = self._train_config
        # assign parameters
        epoch_num = train_config.get('epoch')
        max_epoch = train_config.get('max_epoch')

        # get lr scheduler
        lr_scheduler = LRScheduler.generate_scheduler_by_name(train_config.get('lr_scheduler'), **train_config)
        model_path = train_config.get('model_path')

        if model_path:
            # continue last training
            continue_training = True
            # read corresponding training path
            self._model_folder = os.path.dirname(model_path)
            self._training_folder = os.path.dirname(self._model_folder)
            self._tfb_folder = create_folder(self._training_folder, 'tfbs')

        else:
            # training from scratch
            continue_training = False
            # create model and tensorflow board folder
            time = datetime.datetime.now()
            timestamp = datetime.datetime.strftime(time, '%m%d%H%M%S')
            model_foldername = make_config_string(self._config['model']) + '_' + timestamp

            self._training_folder = create_folder(self._config['base_dir'], model_foldername)
            self._model_folder = create_folder(self._training_folder, 'models')
            self._tfb_folder = create_folder(self._training_folder, 'tfbs')

        return epoch_num, max_epoch, lr_scheduler, continue_training

    def _save_model_with_config(self, sess):
        train_config = self._train_config
        # update model path in train config
        train_config['model_path'] = os.path.join(self._model_folder, 'model-' + str(train_config['epoch']))

        # save model
        self._model_saver.save(sess, train_config['model_path'])
        # save config to yaml file
        config_path = os.path.join(self._model_folder, 'config-' + str(train_config['epoch']) + '.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self._config, f)

    def _update_train_config(self, lr, epoch):
        train_config = self._train_config
        train_config['lr'] = lr
        train_config['epoch'] = epoch

    def _run_epoch(self, sess, data_provider, lr, is_train):
        """
        :param sess:
        :param data_provider:
        :param lr:
        :param is_train:
        :return: [epoch_loss, epoch_pred, epoch_label, epoch_cost_time]
        """
        self._timer.start()
        model = self._model
        if is_train:
            run_func = model.train
        else:
            run_func = model.predict
        loss_list = []
        pred_list = []
        real_list = []
        for batch_data in data_provider.iterate_batch_data():
            loss, pred, real = run_func(sess, batch_data, lr=lr)

            loss_list.append(loss)
            pred_list.append(pred)
            real_list.append(real)

        # shape -> [n_items, horizon, D]
        epoch_preds = concat_arrs_of_dict(pred_list)
        epoch_reals = concat_arrs_of_dict(real_list)

        epoch_avg_loss = np.mean(loss_list)
        # inverse scaling data
        epoch_preds = data_provider.epoch_inverse_scaling(epoch_preds)
        epoch_reals = data_provider.epoch_inverse_scaling(epoch_reals)

        return epoch_avg_loss, epoch_preds, epoch_reals, self._timer.end()
