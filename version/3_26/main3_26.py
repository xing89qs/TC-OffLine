# !/usr/bin/env python
# -- coding:utf-8 --

from solver import *
from global_var import *
import os
from model import *


def prepare_data(dataset_path, train_file_set, test_file_set, item_set, new_load=False):
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    state_listener(start=True)
    state_listener('Building Solver...')
    simple_solver = SimpleSolver(train_file_set, test_file_set, item_set,
                                 train_frame=dataset_path + u'/train_record_frame.csv' if not new_load else None,
                                 test_frame=dataset_path + u'/test_record_frame.csv' if not new_load else None)
    if simple_solver.training_set:
        simple_solver.training_set.record_to_file(dataset_path + '/train_record_frame.csv')
    if simple_solver.test_set:
        simple_solver.test_set.record_to_file(dataset_path + '/test_record_frame.csv')
    state_listener('Extracting Feature...')
    simple_solver.extract_feature(dataset_path)


DATE = u'3_26/'
__MODEL_NAME = u'model.mo'

import pandas as pd
import numpy as np


def cal(x):
    print type(x)
    return sum(x) + 1


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    print np.concatenate((a[1:2], a[-2:]))
    # data = {'A': [1, 1, 2], 'B': [1, 2, 3], 'C': [1, 2, 3,]}
    # frame = pd.DataFrame(data)
    # groups = frame.groupby('A', as_index=False)
    #  print frame.ix[0]['A']

    train_file_set = [[u'date/2014-12-10', u'date/2014-12-11', u'date/2014-12-12', u'date/2014-12-13',
                       u'date/2014-12-14', u'date/2014-12-15', u'date/2014-12-16', ], [u'date/2014-12-17']]
    # train_file_set = [[u'date/2014-12-20', ], [u'date/2014-12-20']]
    test_file_set = [[u'date/2014-12-11', u'date/2014-12-12', u'date/2014-12-13',
                      u'date/2014-12-14', u'date/2014-12-15', u'date/2014-12-16', u'date/2014-12-17'],
                     [u'date/2014-12-18']]
    # test_file_set = [[u'date/2014-12-20', ], [u'date/2014-12-20']]

    check_file_set = [
        [u'date/2014-12-01', u'date/2014-12-02', u'date/2014-12-03', u'date/2014-12-04', u'date/2014-12-05',
         u'date/2014-12-06', u'date/2014-12-07', ], [u'date/2014-12-08']]

    check1_file_set = [
        [u'date/2014-12-10', u'date/2014-12-11', u'date/2014-12-12', u'date/2014-12-13', u'date/2014-12-14',
         u'date/2014-12-15', u'date/2014-12-16', ], [u'date/2014-12-17']]

    target_file_set = [
        [u'date/2014-12-12', u'date/2014-12-13', u'date/2014-12-14', u'date/2014-12-15', u'date/2014-12-16',
         u'date/2014-12-17', u'date/2014-12-18', ], [u'date/2014-12-20']]

    item_file = open(DIRECTORY + u'tianchi_fresh_comp_train_item.csv', 'r')
    item_file.readline()
    item_set = dict()
    for line in item_file:
        line = line.strip().split(',')
        item_set[int(line[0])] = int(line[2])
    # prepare_data(DIRECTORY + DATE + u'dataSet_Train', train_file_set, test_file_set, item_set, new_load=True)
    # prepare_data(DIRECTORY + DATE + u'dataSet_Check', None, check_file_set, item_set, new_load=True)
    # prepare_data(DIRECTORY + DATE + u'dataSet_Check1', None, check1_file_set, item_set, new_load=True)
    # prepare_data(DIRECTORY + DATE + u'dataSet_Target', None, target_file_set, item_set, new_load=True)
    lr = Learning(DIRECTORY + DATE + 'model_4.mo')
    POSITIVIE_SAMPLE_WEIGHT = 1000
    # lr.train(DIRECTORY + DATE + u'dataSet_Train/train_features.csv', DIRECTORY + DATE + 'model_4.mo')
    lr.predict(DIRECTORY + DATE + u'dataSet_Train/test_features.csv',
               DIRECTORY + DATE + u'dataSet_Train/result.txt', item_set)
    lr.predict(DIRECTORY + DATE + u'dataSet_Check1/test_features.csv',
               DIRECTORY + DATE + u'dataSet_Check1/result.txt', item_set)
    lr.predict(DIRECTORY + DATE + u'dataSet_Target/test_features.csv',
               DIRECTORY + DATE + u'dataSet_Target/result.txt', item_set)
