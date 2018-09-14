# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MODEL='/home/kylehsu/experiments/deepcluster/log/miniimagenet/current/checkpoint.pth.tar'
EXP='/home/kylehsu/experiments/deepcluster/log/miniimagenet/current/retrieval'
CONV=5
DATA='/home/kylehsu/store/miniImageNet/train'

python activ-retrieval.py --model ${MODEL} --exp ${EXP} --conv ${CONV} --data ${DATA}
