#!/usr/bin/env python

'''
@author: HYPJUDY 2019/4/15
https://github.com/HYPJUDY

A sample function for feature extraction / classification using
 spatial or temporal network Customize as needed:
e.g. split_set, mode, feature_dim, layer for feature extraction

(general) Input: window_infoFile
(general) Output: feature file corresponding to each line of window_infoFile

Usage:
Please refer to `extract_feature.sh` for multiple gpus and split sets running
e.g.
`python extract_feature.py KnetBn val 1 5 0`
`python extract_feature.py KnetBn test 0 7 6`

Note:
`special_idx` is used because:
the frame number of video_test_0000698/video_test_0000062
is 509/447 (smaller than 512) which need special operation
2666 0, video_test_0000698
949 0, video_test_0000062

This code depends on custom caffe from 
"https://github.com/yjxiong/anet2016-cuhk/" or "https://github.com/yjxiong/caffe"
So install the caffe-action firstly and then export the PYTHONPATH, e.g.
`export PYTHONPATH=/home/administrator/anet2016-cuhk/lib/caffe-action/python:$PYTHONPATH`

This code is referenced from 
https://github.com/yjxiong/caffe/tree/action_recog/action_python
with significant modification.

'''

import caffe
import numpy as np
import math
import cv2
import sys
import os
from os.path import join

feature_path = "/data/THUMOS14/feature/"
mode = 'spatial'  # 'temporal' / 'spatial'

pretrain_dataset = sys.argv[1]
split_set = sys.argv[2]  # 'test' / 'val'
gpu_id = int(sys.argv[3])
split_num = int(sys.argv[4])
split_idx = int(sys.argv[5])

if split_set == 'val':
    win_num = 4741
else:
    win_num = 5103
split_video_num = math.ceil(1.0 * win_num / split_num)
startIdx = min(split_idx * split_video_num, win_num - 1)
endIdx = min((split_idx + 1) * split_video_num, win_num + 1)

'''
The size of each window is `window_size`, we sample it by step `steps`.
Then the number of samples of each window is `window_size / steps`.
In each sample, we use spatial network to extract appearance feature 
with central frame, and we use the output of feature_layer like
"Flatten-673" layer in defined network ResNet as feature.
'''


def video_spatial_prediction(
        vid_name,
        net,
        feature_dim,
        img_dim,
        feature_layer,
        start_frame,
        feature_save_file,
        special_idx,
        window_size=512,
        steps=4,
        mean=[104, 117, 123]
):
    # selection
    num_samples = window_size / steps  # how many frames
    dims = (img_dim, img_dim, 3, num_samples)
    rgb = np.zeros(shape=dims, dtype=np.float64)

    for i in range(num_samples):
        t = i * steps + steps / 2 + start_frame
        if special_idx != 0:
            if t > special_idx:
                t = special_idx
        img_file = join(vid_name,
                        'img_{0:05d}.jpg'.format(t))
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dims[1::-1])
        rgb[:, :, :, i] = img

    # substract mean
    image_mean = np.zeros((img_dim, img_dim, 3))
    image_mean[:, :, :] = np.array(mean)

    rgb = rgb[...] - np.tile(image_mean[..., np.newaxis], (1, 1, 1, rgb.shape[3]))
    rgb = np.transpose(rgb, (1, 0, 2, 3))

    # test
    prediction = np.zeros((rgb.shape[3], feature_dim))

    for i in range(num_samples):
        # (img_dim,img_dim,3) -> (img_dim,img_dim,3,1)
        t = np.reshape(rgb[:, :, :, i], (img_dim, img_dim, 3, 1))
        net.blobs['data'].data[...] = np.transpose(t, (3, 2, 1, 0))
        output = net.forward()
        prediction[i, :] = np.reshape(output[feature_layer], feature_dim)
    np.save(feature_save_file, prediction)


'''
For motion feature, we compute optical flows using number of 
`optical_flow_frames` consecutive frames around the
center frame of a sample, then these optical flows are used
for extracting motion feature with temporal network, where
the output of feature_layer like "global-pool" layer in defined 
network is used as feature.
'''


def video_temporal_prediction(
        vid_name,
        net,
        feature_dim,
        img_dim,
        feature_layer,
        start_frame,
        feature_save_file,
        special_idx,
        window_size=512,
        steps=4,
        optical_flow_frames=5,  # x or y optical flow frame number in a stack
        mean=128
):
    # selection
    num_samples = window_size / steps  # how many stacks
    dims = (img_dim, img_dim, optical_flow_frames * 2, num_samples)
    flow = np.zeros(shape=dims, dtype=np.float64)

    for i in range(num_samples):
        for j in range(optical_flow_frames):
            t = i * steps + j + 1 + start_frame
            if special_idx != 0:
                if t > special_idx:
                    t = special_idx
            flow_x_file = join(vid_name, 'flow_x_{0:05d}.jpg'.format(t))
            flow_y_file = join(vid_name, 'flow_y_{0:05d}.jpg'.format(t))
            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
            img_x = cv2.resize(img_x, dims[1::-1])
            img_y = cv2.resize(img_y, dims[1::-1])

            flow[:, :, j * 2, i] = img_x
            flow[:, :, j * 2 + 1, i] = img_y

    # substract mean
    flow = flow - mean
    flow = np.transpose(flow, (1, 0, 2, 3))  # transpose height and width

    # test
    prediction = np.zeros((num_samples, feature_dim))

    for i in range(num_samples):
        # (img_dim,img_dim,10) -> (img_dim,img_dim,10,1)
        t = np.reshape(flow[:, :, :, i], (img_dim, img_dim, optical_flow_frames * 2, 1))
        net.blobs['data'].data[...] = np.transpose(t, (3, 2, 1, 0))
        output = net.forward()
        # (feature_dim,1,1) -> (feature_dim,)
        prediction[i, :] = np.reshape(output[feature_layer], feature_dim)
    np.save(feature_save_file, prediction)


'''
Use window_infoFile to specify window info, e.g.
512, video_validation_0000417
640, video_validation_0000417
...
'''


def extract_feature(mode, model_def_file, model_file, data_x_path, feature_dim, img_dim, feature_layer):
    # caffe init
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    net = caffe.Net(model_def_file, model_file, caffe.TEST)
    anno_path = '~/Decouple-SSAD/data/' + split_set + '/annotation/'
    window_infoFile = anno_path + 'window_info.log'
    videoPath = '/data/THUMOS14/flow/' + split_set + '/'

    with open(window_infoFile, 'r') as window_infoFile:
        for idx, line in enumerate(window_infoFile):
            # You can use the following line to specify domain
            # [startIdx, endIdx)
            if idx < startIdx:
                continue
            if idx >= endIdx:
                break
            special_idx = 0  # not special
            if split_set == 'test':
                if idx == 949:
                    special_idx = 447
                elif idx == 2666:
                    special_idx = 509

            start_frame, video_name = line.split(',')
            start_frame = int(start_frame)
            video_name = video_name.strip()
            input_video_dir = videoPath + video_name + '/'
            feature_save_file = data_x_path + str(idx) + '.npy'
            if os.path.isfile(feature_save_file):
                print "Exists", idx, line
                continue
            print idx, line
            if mode == 'spatial':
                video_spatial_prediction(input_video_dir,
                                         net, feature_dim, img_dim, feature_layer,
                                         start_frame, feature_save_file, special_idx)
            elif mode == 'temporal':
                video_temporal_prediction(input_video_dir,
                                          net, feature_dim, img_dim, feature_layer,
                                          start_frame, feature_save_file, special_idx)


def main():
    img_dim = 224
    feature_dim = 1024
    data_x_path = join(feature_path, split_set, mode + 'DataX' + pretrain_dataset)
    feature_layer = 'global_pool'
    model_path = '/data/models/'

    if mode == 'spatial':

        if pretrain_dataset == 'UCF101':  # https://github.com/yjxiong/temporal-segment-networks/blob/master/scripts/get_reference_models.sh
            model_def_file = join(model_path, 'ucf101_tsn_reference_bn_inception/tsn_bn_inception_rgb_deploy.prototxt')
            model_file = join(model_path, 'ucf101_tsn_reference_bn_inception/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel')

        elif pretrain_dataset == 'Anet':  # https://github.com/yjxiong/anet2016-cuhk/blob/master/models/get_reference_models.sh
            # pre-trained on ActivityNet v1.3 training set. spatial in ResNet network
            model_def_file = join(model_path, 'anet_pretrained/resnet200_anet_2016_deploy.prototxt')
            model_file = join(model_path, 'anet_pretrained/resnet200_anet_2016.caffemodel')
            feature_dim = 2048
            feature_layer = 'caffe.Flatten_673'

        elif pretrain_dataset == 'KnetBn':  # http://yjxiong.me/others/kinetics_action/
            model_def_file = join(model_path, 'bn_inception_kinetics_pretrained/bn_inception_rgb_deploy.prototxt')
            model_file = join(model_path, 'bn_inception_kinetics_pretrained/bn_inception_kinetics_rgb_pretrained.caffemodel')

        elif pretrain_dataset == 'KnetV3':  # http://yjxiong.me/others/kinetics_action/
            model_def_file = join(model_path, 'inception_v3_kinetics_pretrained/inception_v3_rgb_deploy.prototxt')
            model_file = join(model_path, 'inception_v3_kinetics_pretrained/inception_v3_kinetics_rgb_pretrained.caffemodel')
            feature_dim = 2048
            img_dim = 299

        elif pretrain_dataset == 'Init':
            # pre-trained on ImageNet training set. temporal in BN-Inception network
            model_def_file = join(model_path, 'tsn_bn_inception_init/tsn_bn_inception_rgb_deploy.prototxt')
            model_file = join(model_path, 'tsn_bn_inception_init/bn_inception_rgb_init.caffemodel')

    elif mode == 'temporal':

        if pretrain_dataset == 'UCF101':  # https://github.com/yjxiong/temporal-segment-networks/blob/master/scripts/get_reference_models.sh
            model_def_file = join(model_path, 'ucf101_tsn_reference_bn_inception/tsn_bn_inception_flow_deploy.prototxt')
            model_file = join(model_path, 'ucf101_tsn_reference_bn_inception/ucf101_tsn_flow_reference_bn_inception.caffemodel')

        elif pretrain_dataset == 'Anet':  # https://github.com/yjxiong/anet2016-cuhk/blob/master/models/get_reference_models.sh
            # pre-trained on ActivityNet v1.3 training set. temporal in BN-Inception network
            model_def_file = join(model_path, 'anet_pretrained/bn_inception_anet_2016_temporal_deploy.prototxt')
            model_file = join(model_path, 'anet_pretrained/bn_inception_anet_2016_temporal.caffemodel.v5')

        elif pretrain_dataset == 'KnetBn':  # http://yjxiong.me/others/kinetics_action/
            model_def_file = join(model_path, 'bn_inception_kinetics_pretrained/bn_inception_flow_deploy.prototxt')
            model_file = join(model_path, 'bn_inception_kinetics_pretrained/bn_inception_kinetics_flow_pretrained.caffemodel')

        elif pretrain_dataset == 'KnetV3':  # http://yjxiong.me/others/kinetics_action/
            model_def_file = join(model_path, 'inception_v3_kinetics_pretrained/inception_v3_flow_deploy.prototxt')
            model_file = join(model_path, 'inception_v3_kinetics_pretrained/inception_v3_flow_kinetics.caffemodel')
            feature_dim = 2048
            img_dim = 299

        elif pretrain_dataset == 'Init':
            # pre-trained on ImageNet training set. temporal in BN-Inception network
            model_def_file = join(model_path, 'tsn_bn_inception_init/tsn_bn_inception_flow_deploy.prototxt')
            model_file = join(model_path, 'tsn_bn_inception_init/bn_inception_flow_init.caffemodel')

    if not os.path.exists(data_x_path):
        os.makedirs(data_x_path)
    extract_feature(mode, model_def_file, model_file, data_x_path, feature_dim, img_dim, feature_layer)


if __name__ == "__main__":
    main()
