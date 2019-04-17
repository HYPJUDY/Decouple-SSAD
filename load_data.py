# -*- coding: utf-8 -*-
"""
@author: HYPJUDY 2019/4/15
https://github.com/HYPJUDY

Decoupling Localization and Classification in Single Shot Temporal Action Detection
-----------------------------------------------------------------------------------

Functions to get train and test data

"""


import numpy as np
import random
from os.path import join
import pickle
from config import get_anno_ath, get_data_x_path
import sys

small_num_data_for_test = False


def read_window_info(path):
    result = []
    with open(path, 'r') as path:
        for line in path:
            sFrame, video_name = line.split(',')
            result.append([int(sFrame), video_name.strip()])
    return result


def read_pickle(path):
    with open(path, 'rb') as path:
        if sys.version_info[0] == 2:  # python2
            result = pickle.load(path)
        else:  # python3
            result = pickle.load(path, encoding='bytes')
    return result


############################# GET TRAIN DATA ##############################

def batch_data_process(batch_data):
    batch_size = len(batch_data)
    new_batch_data = np.array(np.ones([1, batch_data[0].shape[1]]))
    batch_start_index = [0]
    for i in range(batch_size):
        new_batch_data = np.concatenate((new_batch_data, batch_data[i]))
        if i < (batch_size - 1):
            batch_start_index.append(batch_start_index[-1] + len(batch_data[i]))
    new_batch_data = new_batch_data[1:]
    batch_start_index.append(len(new_batch_data))
    return new_batch_data, np.array(batch_start_index)


def get_train_data(config, mode, pretrain_dataset, shuffle=True):
    batch_size = config.batch_size
    split_set = config.train_split_set
    data_x_path = get_data_x_path(config.feature_path, split_set, mode, pretrain_dataset)
    anno_path = get_anno_ath(split_set)
    # Since the dataX is matched with window_info.log,
    # window_info need to load from pre-defined file
    gt_label_file = join(anno_path, 'gt_label.pkl')
    gt_info_file = join(anno_path, 'gt_info.pkl')
    gt_label = read_pickle(gt_label_file)
    gt_info = read_pickle(gt_info_file)

    if not small_num_data_for_test:
        num_data = len(gt_label)
    else:
        num_data = batch_size

    batch_dataX = []
    batch_gt_label = []
    batch_gt_info = []
    batch_index = []

    batch_start_list = [i * batch_size for i in range(int(num_data / batch_size))]
    if (num_data - (batch_start_list[-1] + batch_size)) > (batch_size / 8):
        batch_start_list.append(num_data - batch_size)

    batch_shuffle_list = list(range(num_data))

    if shuffle:
        random.seed(6)
        random.shuffle(batch_shuffle_list)

    for bstart in batch_start_list:
        data_list = batch_shuffle_list[bstart:(bstart + batch_size)]
        tmp_batch_dataX = []
        tmp_batch_gt_label = []
        tmp_batch_gt_info = []
        for idx in data_list:
            adataX = np.load(join(data_x_path, str(idx) + '.npy'))
            tmp_batch_dataX.append(adataX)
            tmp_batch_gt_label.append(gt_label[idx])
            tmp_batch_gt_info.append(gt_info[idx])

        batch_dataX.append(np.array(tmp_batch_dataX))

        tmp_batch_gt_label, start_index = batch_data_process(tmp_batch_gt_label)
        batch_gt_label.append(tmp_batch_gt_label)
        batch_index.append(start_index)

        tmp_batch_gt_info, start_index = batch_data_process(tmp_batch_gt_info)
        batch_gt_info.append(tmp_batch_gt_info)

    return batch_dataX, batch_gt_label, batch_gt_info, batch_index


############################# GET TEST DATA ##############################

def get_test_data(config, mode, pretrain_dataset):
    batch_size = config.batch_size
    split_set = config.test_split_set
    data_x_path = get_data_x_path(config.feature_path, split_set, mode, pretrain_dataset)
    anno_path = get_anno_ath(split_set)

    # Since the dataX is matched with window_info.log,
    # window_info need to load from pre-defined file
    window_info_path = join(anno_path, 'window_info.log')
    window_info = read_window_info(window_info_path)

    if not small_num_data_for_test:
        num_data = len(window_info)
    else:
        num_data = batch_size

    batch_dataX = []
    batch_window_info = []

    batch_start_list = [i * batch_size for i in range(int(num_data / batch_size))]
    if (num_data - (batch_start_list[-1] + batch_size)) > (batch_size / 8):
        batch_start_list.append(num_data - batch_size)

    batch_list = list(range(num_data))

    for bstart in batch_start_list:
        data_list = batch_list[bstart:(bstart + batch_size)]
        tmp_batch_dataX = []
        tmp_batch_window_info = []
        for idx in data_list:
            adataX = np.load(join(data_x_path, str(idx) + '.npy'))
            tmp_batch_dataX.append(adataX)
            tmp_batch_window_info.append(window_info[idx])

        batch_dataX.append(np.array(tmp_batch_dataX))
        batch_window_info.append(tmp_batch_window_info)

    return batch_dataX, batch_window_info
