#!/usr/bin/env bash

DIR="/home/kylehsu/store/miniImageNet/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=500
WORKERS=12
EXP="/home/kylehsu/experiments/deepcluster/log/miniimagenet/current"
PYTHON="python2"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --verbose --workers ${WORKERS} \
  --reassign 10 --start_epoch 50 \
  --resume "/home/kylehsu/experiments/deepcluster/log/miniimagenet/current/checkpoints/checkpoint_3.pth.tar"
