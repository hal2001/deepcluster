# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MODEL='/home/kylehsu/experiments/deepcluster/log/miniimagenet/current/checkpoint.pth.tar'
ARCH='alexnet'
EXP='/home/kylehsu/experiments/deepcluster/log/miniimagenet/current/gradient_ascent'
CONV=5

python gradient_ascent.py --model ${MODEL} --exp ${EXP} --conv ${CONV} --arch ${ARCH} \
--idim 64 --lr 0.3
