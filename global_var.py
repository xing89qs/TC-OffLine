#!/usr/bin/env python
# -- coding:utf-8 --

import time

# 数据目录
DIRECTORY = u'/Volumes/disk1s2/天池数据/fresh_comp_offline/'

#  用户行为
BEHAVIOR_NUM = 4
BROWSE = 0
FAVOR = 1
ADD_TO_CART = 2
BUY = 3


# 正样本权重
POSITIVIE_SAMPLE_WEIGHT = 100


def str2time_stamp(_str, re='%Y-%m-%d %H'):
    return int(time.mktime(time.strptime(_str, re)))


def state_listener(msg='Starting...', start=False):
    global start_stamp
    if start:
        start_stamp = time.clock()
    cost_time = time.clock() - start_stamp
    print '经过了 %d s : ' % cost_time, msg
