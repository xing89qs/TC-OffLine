#!/usr/bin/env python
# -- coding:utf-8 --

from global_var import *
from pandas import DataFrame, read_csv
import os


class DataSet(object):
    def __init__(self, file_list, label_file_list, item_set, data_frame=None, is_train_set=True):
        self.is_train_set = is_train_set
        self.item_set = item_set
        if data_frame and os.path.exists(data_frame):
            self.record_frame = read_csv(data_frame)
        else:
            state_listener('Reading Label...')
            _cnt = 0
            for _file in label_file_list:
                label_file = open(DIRECTORY + _file + '.txt')
                self.buy_list = set()
                for line in label_file:
                    _record = [_ for _ in line.strip().split(',')]
                    if int(_record[2]) - 1 == BUY:
                        self.buy_list.add((int(_record[0]), int(_record[1])))
                    if _cnt % 1000000 == 0:
                        state_listener('Reading %05d' % _cnt)
                    _cnt += 1
            label_file.close()

            record_list = []
            state_listener('Reading Record...')
            _cnt = 0
            for _file in file_list:
                data_file = open(DIRECTORY + _file + u'.txt', 'r')
                for line in data_file:
                    record = [_ for _ in line.strip().split(',')]
                    if (int(record[0]), int(record[1])) in self.buy_list:
                        record.append('1')
                    else:
                        record.append('0')
                    DataSet.__to_int(record)
                    record_list.append(tuple(record))
                    if _cnt % 1000000 == 0:
                        state_listener('Reading %05d' % _cnt)
                    _cnt += 1
                data_file.close()
            self.record_frame = DataFrame(record_list,
                                          columns=['uid', 'iid', 'behavior', 'geohash', 'category', 'time', 'label'])

    @staticmethod
    def __to_int(record):
        INT_INDEXS = [0, 1, 2, 4, 6]  # 转化为int 节省空间
        for _ in INT_INDEXS:
            record[_] = int(record[_])
        record[2] -= 1  # 行为变成0开始标号
        record[5] = str2time_stamp(record[5])

    def __is_record_valid(self, record):
        return int(record[1]) in self.item_set

    def record_to_file(self, path):
        self.record_frame.to_csv(path)
