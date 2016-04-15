#!/usr/bin/env python
# -- coding:utf-8 --

from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import *
from sklearn.externals import joblib
from global_var import *


class Learning1(object):
    def __init__(self, model_file=''):
        if model_file:
            self.model = joblib.load(model_file)
        else:
            self.model = GradientBoostingRegressor()
        self.scaler = StandardScaler()
        pass

    def train(self, filename, model_name):
        _file = open(filename)
        train_x = []
        train_y = []
        for _ in _file:
            vec = [x for x in _.strip().split(',')]
            train_x.append(map(int, vec[:-1]))
            train_y.append(map(int, vec[-1]))
        _file.close()

        self.scaler.fit(train_x)
        train_x = self.scaler.transform(train_x)
        f = lambda x: POSITIVIE_SAMPLE_WEIGHT if x else 1
        weight = np.array([f(_) for _ in train_y])
        self.model.fit(train_x, train_y, weight)
        joblib.dump(self.model, model_name)

    def predict(self, filename, result_file_name, item_set, is_test=True):
        _file = open(filename)
        test_x = []
        test_y = []
        for _ in _file:
            vec = [x for x in _.strip().split(',')]
            test_x.append(map(int, vec[:-1]))
            test_y.append(int(vec[-1]))
        _file.close()
        self.scaler.fit(test_x)
        test_x = self.scaler.transform(test_x)
        predict_y = self.model.predict(test_x)

        tot = sum(test_y)
        print tot

        if tot == 0:
            tot = 10000000000

        ouput_file = open(result_file_name, 'w')

        for num in xrange(50, 1000, 50):
            temp = np.argpartition(-predict_y, num)
            result_args = temp[:num]
            cnt = 0
            for _ in result_args:
                if test_y[_] == 1:
                    cnt += 1
            precision = cnt / float(num)
            recall = cnt / float(tot)

            if cnt == 0:
                f = 0
            else:
                f = 2 * precision * recall / (precision + recall)
            if is_test:
                print 'num = %d' % num, precision, recall, f

        ouput_file.close()
