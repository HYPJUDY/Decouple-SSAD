# -*-coding:utf-8-*-

'''
@author: HYPJUDY 2019/4/15
https://github.com/HYPJUDY

Get video ground truth annotation files
based on SSN: https://github.com/yjxiong/action-detection
the ground truth is from SSN code

Usage: put this code file under the root of SSN code to run
'''

from proposal2feature import SSNDataSet
import pandas as pd

SSN_root_path = '~/SSN/'
train_prop_file = SSN_root_path + 'thumos14_tag_proposal_list/thumos14_tag_val_proposal_list.txt'
test_prop_file = SSN_root_path + 'thumos14_tag_proposal_list/thumos14_tag_test_proposal_list.txt'

classList = [0, 7, 9, 12, 21, 22, 23, 24, 26, 31, 33, 36, 40, 45, 51, 68, 79, 85, 92, 93, 97]
typeList = ['Ambiguous', 'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
            'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
            'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
            'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']

########################### test video #############################
testData = SSNDataSet("", test_prop_file, test_mode=True)

video_names = []
video_names_num = []
type_idx = []
type = []
startFrame = []
endFrame = []
frame_num = []

for video in testData.video_list:
    video_name = video.id.split('/')[-1]
    video_name_num = int(video_name.split('_')[-1])
    num_frames = video.num_frames
    for gt in video.gt:
        video_names.append(video_name)
        video_names_num.append(video_name_num)
        type.append(typeList[gt.label])
        type_idx.append(classList[gt.label])
        startFrame.append(gt.start_frame)
        endFrame.append(gt.end_frame)
        frame_num.append(num_frames)

dic = {}
dic['video'] = video_names
dic['video_name_num'] = video_names_num
dic['type'] = type
dic['type_idx'] = type_idx
dic['startFrame'] = startFrame
dic['endFrame'] = endFrame
dic['frame_num'] = frame_num

df = pd.DataFrame(dic, columns=['video', 'video_name_num',
                                'type', 'type_idx', 'startFrame', 'endFrame', 'frame_num'])
df.sort_values(['video_name_num', 'type_idx', 'startFrame'], inplace=True)
df = df.drop('video_name_num', axis=1)
df.to_csv('thumos14_test_annotation.csv', encoding='utf-8', index=False)

######################## training video ##########################
trainingData = SSNDataSet("", train_prop_file, test_mode=False)

video_names = []
video_names_num = []
type_idx = []
type = []
startFrame = []
endFrame = []
frame_num = []

for video in trainingData.video_list:
    video_name = video.id.split('/')[-1]
    video_name_num = int(video_name.split('_')[-1])
    num_frames = video.num_frames
    for gt in video.gt:
        video_names.append(video_name)
        video_names_num.append(video_name_num)
        type.append(typeList[gt.label])
        type_idx.append(classList[gt.label])
        startFrame.append(gt.start_frame)
        endFrame.append(gt.end_frame)
        frame_num.append(num_frames)

dic = {}
dic['video'] = video_names
dic['video_name_num'] = video_names_num
dic['type'] = type
dic['type_idx'] = type_idx
dic['startFrame'] = startFrame
dic['endFrame'] = endFrame
dic['frame_num'] = frame_num

df = pd.DataFrame(dic, columns=['video', 'video_name_num',
                                'type', 'type_idx', 'startFrame', 'endFrame', 'frame_num'])
df.sort_values(['video_name_num', 'type_idx', 'startFrame'], inplace=True)
df = df.drop('video_name_num', axis=1)
df.to_csv('thumos14_val_annotation.csv', encoding='utf-8', index=False)
