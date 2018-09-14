#!/bin/bash

MODEL='/home/kylehsu/experiments/deepcluster/log/celeba/current/checkpoint.pth.tar'
EXP='/home/kylehsu/experiments/deepcluster/log/celeba/current/features'
DATA='/home/kylehsu/store/maml/data/celeba/cropped/Img_resized84_split'

python infer.py --model ${MODEL} --exp ${EXP} --data ${DATA} --dataset celeba
