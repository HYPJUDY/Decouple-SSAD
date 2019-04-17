# !/bin/bash

mkdir -p logs
mkdir -p results
mkdir -p models
echo "start running"
SECONDS=0

#################### script for train/test #######################

stage=train_test_fuse  # train/test/fuse/train_test_fuse
pretrain_dataset=UCF101  # UCF101/KnetV3
mode=temporal  # temporal/spatial
file_name=decouple_ssad
method=decouple_ssad
method_temporal=decouple_ssad
gpu="0"
if [ $stage == train_test_fuse ]
then
    LOG_FILE=logs/${mode}_${pretrain_dataset}_${method}.log
    CUDA_VISIBLE_DEVICES=${gpu} python -u ${file_name}.py ${stage} ${pretrain_dataset} ${mode} ${method} ${method_temporal} > ${LOG_FILE} &
else
    CUDA_VISIBLE_DEVICES=${gpu} python -u ${file_name}.py ${stage} ${pretrain_dataset} ${mode} ${method} ${method_temporal} &
fi

mode=spatial
gpu="1"
if [ $stage == train_test_fuse ]
then
   LOG_FILE=logs/${mode}_${pretrain_dataset}_${method}.log
   CUDA_VISIBLE_DEVICES=${gpu} python -u ${file_name}.py ${stage} ${pretrain_dataset} ${mode} ${method} ${method_temporal} > ${LOG_FILE} &
else
   CUDA_VISIBLE_DEVICES=${gpu} python -u ${file_name}.py ${stage} ${pretrain_dataset} ${mode} ${method} ${method_temporal} &
fi

wait
# # ####################### script for fuse ##########################
echo "finish training/testing, start fusing"
stage=fuse
python -u ${file_name}.py ${stage} ${pretrain_dataset} ${mode} ${method} ${method_temporal}


tRun=$SECONDS
echo "$(($tRun / 60)) minutes and $(($tRun % 60)) seconds elapsed for running."
