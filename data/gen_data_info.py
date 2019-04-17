#!/usr/bin/env python

'''
@author: HYPJUDY 2019/4/15
https://github.com/HYPJUDY

Construct training and testing data similar to SSAD:
slide window through each untrimmed video with overlap.

(general) Input: thumos14 val and test annotation csv files
(general) Output: sliding window information of validation set and test set.
                  label and ground truth information of validation set.

Usage:
`python gen_data_info.py`

'''


import os
import pickle
import sys
from os.path import join

import numpy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, get_anno_ath, get_anno_df


def window_data(start, anno_df, video_name, config):
    start_frame = start
    end_frame = start_frame + config.window_size
    label_info = []
    box_info = []
    window_info = [int(start_frame), video_name]
    class_real = [0] + config.class_real  # num_classes + 1
    for i in range(len(anno_df)):
        astart = anno_df.startFrame.values[i]
        aend = anno_df.endFrame.values[i]
        overlap = min(end_frame, aend) - max(astart, start_frame)
        overlap_ratio = float(overlap) / (aend - astart)
        if overlap_ratio > config.overlap_ratio_threshold:
            # the overlap region is corrected
            corrected_start = max(astart, start_frame) - start_frame
            corrected_end = min(aend, end_frame) - start_frame
            one_hot = [0] * config.num_classes
            one_hot[class_real.index(anno_df.type_idx.values[i])] = 1
            label_info.append(one_hot)
            box_info.append([float(corrected_start) / config.window_size,
                             float(corrected_end) / config.window_size, overlap_ratio])

    box_info = numpy.array(box_info)
    box_info = box_info.astype('float32')
    label_info = numpy.array(label_info)
    return label_info, box_info, window_info


def slinding_window(anno_df, video_name, config, is_train=True):
    window_size = config.window_size
    video_anno_df = anno_df[anno_df.video == video_name]
    frame_count = video_anno_df.frame_num.values[0]
    len_df = frame_count - 9
    if is_train:
        stride = window_size / 4
    else:
        # bigger step to improve efficiency for testing
        stride = window_size / 2

    n_window = int((len_df + stride - window_size) / stride)
    windows_start = [i * stride for i in range(n_window)]
    if is_train and n_window == 0:
        windows_start = [0]
    if len_df > window_size:
        windows_start.append(len_df - window_size)
    label = []
    box_info = []
    window_info = []
    for start in windows_start:
        if is_train:  # for training data construction
            # only keep windows containing at least one ground truth instance.
            tmp_label, tmp_box, tmp_window_info = \
                window_data(start, video_anno_df, video_name, config)
            if len(tmp_label) > 0:
                label.append(tmp_label)
                box_info.append(tmp_box)
                window_info.append(tmp_window_info)
        else:
            window_info.append([int(start), video_name])

    return label, box_info, window_info


def video_process(anno_df, config, is_train=True):
    video_name_list = list(set(anno_df.video.values[:].tolist()))  # sorted, unique
    label = []
    boxes = []
    window_info = []

    for video_name in video_name_list:
        tmplabel, tmpboxes, tmp_window_info = \
            slinding_window(anno_df, video_name, config, is_train)

        if is_train and (len(tmplabel) > 0):
            label.extend(tmplabel)
            boxes.extend(tmpboxes)
            window_info.extend(tmp_window_info)
        else:
            window_info.extend(tmp_window_info)

    return label, boxes, window_info


if __name__ == "__main__":
    config = Config()

    # ------ val ---------
    split_set = config.train_split_set
    anno_path = get_anno_ath(split_set)
    anno_df = get_anno_df(anno_path, split_set)

    gt_label, gt_info, gt_window_info = video_process(anno_df, config, True)
    with open(join(anno_path, 'gt_label.pkl'), 'wb') as fw:
        pickle.dump(gt_label, fw)
    with open(join(anno_path, 'gt_info.pkl'), 'wb') as fw:
        pickle.dump(gt_info, fw)

    with open(join(anno_path, 'window_info.log'), 'w') as fw:
        fw.writelines("%d, %s\n" % (window_info[0], window_info[1]) for window_info in gt_window_info)

    # ------ test ---------
    split_set = config.test_split_set
    anno_path = get_anno_ath(split_set)
    anno_df = get_anno_df(anno_path, split_set)

    _, _, gt_window_info = video_process(anno_df, config, False)
    with open(join(anno_path, 'window_info.log'), 'w') as fw:
        fw.writelines("%d, %s\n" % (window_info[0], window_info[1]) for window_info in gt_window_info)
