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


DATE = u'3_28/'

import pandas as pd
import numpy as np


def f(x):
    if x['A'] <= x['B']:
        return True
    return False


if __name__ == '__main__':
    data = {'A': [1, 1, 2], 'B': [1, 2, 3], 'C': [1, 2, 3, ]}
    frame = pd.DataFrame(data)
    frame = frame[frame.apply(f, axis=1)]
    for _ in frame.values:
        print _
    print frame

    train_file_set = [[u'date/2014-12-10', u'date/2014-12-11', u'date/2014-12-12', u'date/2014-12-13',
                       u'date/2014-12-14', u'date/2014-12-15', u'date/2014-12-16', ], [u'date/2014-12-17']]
    train1_file_set = [[u'date/2014-12-08', u'date/2014-12-09', u'date/2014-12-10', u'date/2014-12-11',
                        u'date/2014-12-12', u'date/2014-12-13', u'date/2014-12-14', ], [u'date/2014-12-15']]
    train2_file_set = [[u'date/2014-12-07', u'date/2014-12-08', u'date/2014-12-09', u'date/2014-12-10',
                        u'date/2014-12-11', u'date/2014-12-12', u'date/2014-12-13', ], [u'date/2014-12-14']]
    train3_file_set = [[u'date/2014-12-06', u'date/2014-12-07', u'date/2014-12-08', u'date/2014-12-09',
                        u'date/2014-12-10', u'date/2014-12-11', u'date/2014-12-12', ], [u'date/2014-12-13']]
    train4_file_set = [[u'date/2014-12-04', u'date/2014-12-05', u'date/2014-12-06', u'date/2014-12-07',
                        u'date/2014-12-08', u'date/2014-12-09', u'date/2014-12-10', ], [u'date/2014-12-11']]

    # train_file_set = [[u'date/2014-12-20', ], [u'date/2014-12-20']]
    test_file_set = [[u'date/2014-12-11', u'date/2014-12-12', u'date/2014-12-13',
                      u'date/2014-12-14', u'date/2014-12-15', u'date/2014-12-16', u'date/2014-12-17'],
                     [u'date/2014-12-18']]
    # test_file_set = [[u'date/2014-12-20', ], [u'date/2014-12-20']]

    check_file_set = [
        [u'date/2014-12-01', u'date/2014-12-02', u'date/2014-12-03', u'date/2014-12-04', u'date/2014-12-05',
         u'date/2014-12-06', u'date/2014-12-07', ], [u'date/2014-12-08']]

    check1_file_set = [
        [u'date/2014-12-09', u'date/2014-12-10', u'date/2014-12-11', u'date/2014-12-12', u'date/2014-12-13',
         u'date/2014-12-14', u'date/2014-12-15', ], [u'date/2014-12-16']]

    target_file_set = [
        [u'date/2014-12-12', u'date/2014-12-13', u'date/2014-12-14', u'date/2014-12-15', u'date/2014-12-16',
         u'date/2014-12-17', u'date/2014-12-18', ], [u'date/2014-12-20']]

    item_file = open(DIRECTORY + u'tianchi_fresh_comp_train_item.csv', 'r')
    item_file.readline()
    item_set = dict()
    for line in item_file:
        line = line.strip().split(',')
        item_set[int(line[0])] = int(line[2])
    # prepare_data(DIRECTORY + DATE + u'dataSet_Train', train_file_set, test_file_set, item_set)
    # prepare_data(DIRECTORY + DATE + u'dataSet_Train1', train1_file_set, None, item_set)
    # prepare_data(DIRECTORY + DATE + u'dataSet_Train2', train2_file_set, None, item_set)
    # prepare_data(DIRECTORY + DATE + u'dataSet_Train3', train3_file_set, None, item_set)
    # prepare_data(DIRECTORY + DATE + u'dataSet_Train4', train4_file_set, None, item_set)
    # prepare_data(DIRECTORY + DATE + u'dataSet_Check', None, check_file_set, item_set)
    # prepare_data(DIRECTORY + DATE + u'dataSet_Check1', None, check1_file_set, item_set)
    # prepare_data(DIRECTORY + DATE + u'dataSet_Target', None, target_file_set, item_set)

    # lr = Learning(DIRECTORY+DATE+'model2.mo')

    lr = Learning()

    lr.train([DIRECTORY + DATE + u'dataSet_Train/train_features/', DIRECTORY + DATE + u'dataSet_Train1/train_features/',
              DIRECTORY + DATE + u'dataSet_Train2/train_features/',
              DIRECTORY + DATE + u'dataSet_Train3/train_features/',
              DIRECTORY + DATE + u'dataSet_Train4/train_features/', ]
             , DIRECTORY + DATE + 'model4.mo', item_set)

    #lr.train([DIRECTORY + DATE + u'dataSet_Train/train_features/'], DIRECTORY + DATE + 'model1.mo', item_set)
    lr.predict(DIRECTORY + DATE + u'dataSet_Train/test_features/',
               DIRECTORY + DATE + u'dataSet_Train/result.txt', item_set)
    lr.predict(DIRECTORY + DATE + u'dataSet_Check/test_features/',
               DIRECTORY + DATE + u'dataSet_Check/result.txt', item_set)
    lr.predict(DIRECTORY + DATE + u'dataSet_Check1/test_features/',
               DIRECTORY + DATE + u'dataSet_Check1/result.txt', item_set)
    lr.predict(DIRECTORY + DATE + u'dataSet_Target/test_features/',
               DIRECTORY + DATE + u'dataSet_Target/result.txt', item_set)
