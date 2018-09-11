# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MODEL='/home/kylehsu/experiments/deepcluster/log/miniimagenet/vanilla/checkpoint.pth.tar'
EXP='/home/kylehsu/experiments/deepcluster/log/miniimagenet/features'
DATA='/home/kylehsu/store/miniImageNet'

python infer.py --model ${MODEL} --exp ${EXP} --data ${DATA}
