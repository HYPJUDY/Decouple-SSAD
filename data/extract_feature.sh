# !/bin/bash

cd Decouple-SSAD/code
#################### script for extracting feature #######################
pretrain_dataset=UCF101
split_set=val
gpu_id=0
split_num=3
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 0 &
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 1 &
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 2 &

pretrain_dataset=UCF101
split_set=test
gpu_id=0
split_num=3
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 0 &
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 1 &
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 2 &

pretrain_dataset=KnetV3
split_set=val
gpu_id=1
split_num=3
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 0 &
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 1 &
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 2 &

pretrain_dataset=KnetV3
split_set=test
gpu_id=1
split_num=3
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 0 &
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 1 &
python extract_feature.py ${pretrain_dataset} ${split_set} ${gpu_id} ${split_num} 2 &
