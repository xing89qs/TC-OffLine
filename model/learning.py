#!/usr/bin/env python
# -- coding:utf-8 --

from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import *
from sklearn.externals import joblib
from global_var import *
import math
from random import randint


class Learning(object):
    def __init__(self, model_file=''):
        if model_file:
            self.model1 = joblib.load(model_file + '1')
            # self.model2 = joblib.load(model_file + '2')
            # self.model3 = joblib.load(model_file + '3')
            # self.model4 = joblib.load(model_file + '4')
        else:
            self.model1 = GradientBoostingRegressor(n_estimators=100, max_depth=3)
            # self.model2 = LogisticRegression(n_jobs=-1, max_iter=100000, class_weight={0: 1.0 / 10, 1: 9.0 / 10})
            # self.model2 = LogisticRegression(n_jobs=-1, max_iter=100000, class_weight={0: 1.0 / 10, 1: 9.0 / 10})
            # self.model3 = RandomForestRegressor(n_estimators=100)
            # self.model4 = GradientBoostingRegressor()
        self.scaler = StandardScaler()
        pass

    __FEATURES = [
        u'_ui_brow_count', u'_ui_brow_max_time', u'_ui_fav_count', u'_ui_fav_max_time',
        u'_ui_atc_count', u'_ui_atc_max_time', u'_ui_buy_count',
        u'_ui_buy_max_time', u'_u_brow_count', u'_u_brow_max_time', u'_u_fav_count',
        u'_u_fav_max_time', u'_u_atc_count', u'_u_atc_max_time', u'_u_buy_count',
        u'_u_buy_max_time', u'_i_brow_count', u'_i_brow_max_time', u'_i_fav_count',
        u'_i_fav_max_time', u'_i_atc_count', u'_i_atc_max_time', u'_i_buy_count',
        u'_i_buy_max_time', u'_uc_brow_count', u'_uc_brow_max_time',
        u'_uc_fav_count', u'_uc_fav_max_time', u'_uc_atc_count',
        u'_uc_atc_max_time', u'_uc_buy_count', u'_uc_buy_max_time',
    ]

    @staticmethod
    def merge(frame_list, left, right):
        if left == right:
            return frame_list[left]
        mid = (left + right) / 2
        if left + 1 != right:
            l_frame = Learning.merge(frame_list, left, mid)
            r_frame = Learning.merge(frame_list, mid + 1, right)
        else:
            l_frame = frame_list[left]
            r_frame = frame_list[right]
        ret = pd.merge(l_frame, r_frame, how='outer', on=['uid', 'iid'])
        del l_frame
        del r_frame
        return ret

    def __has_act(self, x):
        return x['iid'] in self.item_set

    def train(self, feature_folders, model_name,item_set):
        self.item_set = item_set
        train_x = []
        train_y = []
        for feature_folder in feature_folders:
            frame_list = [pd.read_csv(feature_folder + _ + '.csv', index_col=0) for _ in self.__FEATURES]
            train_feature_frame = Learning.merge(frame_list, 0, len(frame_list) - 1)
            label_frame = pd.read_csv(feature_folder + 'label.csv', index_col=0)
            train_feature_frame = pd.merge(label_frame, train_feature_frame, how='outer', on=['uid', 'iid'])
            train_feature_frame = train_feature_frame[train_feature_frame.apply(self.__has_act, axis=1)]
            #train_feature_frame.reindex(xrange(len(train_feature_frame)))
            for _ in train_feature_frame.values:
                tmp = _[3:]
                if int(_[2]) == 1 or randint(0, 100) <= 3:
                    train_x.append(map(int, tmp))
                    train_y.append(int(_[2]))
            del train_feature_frame
        print sum(train_y)
        print len(train_x)

        self.scaler.fit(train_x)
        train_x = self.scaler.transform(train_x)
        # POSITIVIE_SAMPLE_WEIGHT = 1000
        f = lambda x: POSITIVIE_SAMPLE_WEIGHT if x else 1
        weight = np.array([f(_) for _ in train_y])
        self.model1.fit(train_x, train_y)
        joblib.dump(self.model1, model_name + '1')
        # self.model2.fit(train_x, train_y)
        # joblib.dump(self.model2, model_name + '2')
        # self.model3.fit(train_x, train_y, weight)
        # joblib.dump(self.model3, model_name + '3')
        # predict_y1 = self.model1.predict(train_x)
        # predict_y2 = self.model2.predict(train_x)
        # predict_y3 = self.model3.predict(train_x)
        # new_x = map(list, zip(predict_y1, predict_y2, predict_y3))
        # self.model4.fit(new_x, train_y)
        # joblib.dump(self.model4, model_name + '4')

    def __has_buy(self, x):
        tmp = min(x['ui_brow_max_time'], x['ui_fav_max_time'], x['ui_atc_max_time'])
        if (tmp > 0 and x['ui_buy_max_time'] > 0) and (tmp < 86400 and x['ui_buy_max_time'] < 86400):
            if tmp >= x['ui_buy_max_time']:
                return False
        return True

    def predict(self, feature_folder, result_file_name, item_set, is_test=True):
        frame_list = [pd.read_csv(feature_folder + _ + '.csv', index_col=0) for _ in self.__FEATURES]
        test_feature_frame = Learning.merge(frame_list, 0, len(frame_list) - 1)
        label_frame = pd.read_csv(feature_folder + 'label.csv', index_col=0)
        test_feature_frame = pd.merge(label_frame, test_feature_frame, how='outer', on=['uid', 'iid'])
        test_feature_frame = test_feature_frame[test_feature_frame.apply(self.__has_buy, axis=1)]
        test_feature_frame.reindex(xrange(len(test_feature_frame)))
        print len(test_feature_frame)

        test_x = []
        test_y = []
        print test_feature_frame.columns
        for _ in test_feature_frame.values:
            tmp = _[3:]
            test_x.append(map(int, tmp))
            test_y.append(int(_[2]))
        tot = sum(test_y)
        print tot
        self.scaler.fit(test_x)
        test_x = self.scaler.transform(test_x)
        '''
        predict_y1 = self.model1.predict(test_x)
        predict_y2 = self.model2.predict(test_x)
        predict_y3 = self.model3.predict(test_x)
        new_x = map(list, zip(predict_y1, predict_y2, predict_y3))
        '''
        predict_y = self.model1.predict(test_x)
        print sum(predict_y == 1)

        if tot == 0:
            tot = 10000000000

        ouput_file = open(result_file_name, 'w')

        for num in xrange(50, 1000, 50):
            temp = np.argpartition(-predict_y, num)
            result_args = temp[:num]
            cnt = 0
            for _ in result_args:
                if num == 600:
                    ouput_file.write((str(test_feature_frame.iloc[_]['uid']) + ',' + str(
                        test_feature_frame.iloc[_]['iid']) + '\n').encode('UTF-8'))
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
