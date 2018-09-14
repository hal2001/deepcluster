#!/usr/bin/env bash

DIR="/home/kylehsu/store/maml/data/celeba/cropped/Img_resized84_split/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=2000
WORKERS=12
EXP="/home/kylehsu/experiments/deepcluster/log/celeba/current"
PYTHON="python2"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} \
  --reassign 2 --dataset celeba
