#!/usr/bin/env python
# -- coding:utf-8 --


from dataset import DataSet
from global_var import *
from feature_extraction import *


class Solver(object):
    def __init__(self):
        pass

    def __load_data(self):
        pass

    def extract_feature(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def get_f_score(self):
        pass


class SimpleSolver(Solver):
    def __init__(self, train_file_set, test_file_set, item_set, train_frame=None, test_frame=None):
        Solver.__init__(self)
        self.train_file_set = train_file_set
        self.test_file_set = test_file_set
        self.__load_data(train_file_set, test_file_set, item_set, train_frame, test_frame)

    def __load_data(self, train_file_set, test_file_set, item_set, train_frame, test_frame):
        state_listener('Loading Items...')
        self.item_set = item_set
        self.training_set = None
        if train_file_set:
            state_listener('Loading Training Set...')
            self.training_set = DataSet(train_file_set[0], train_file_set[1], self.item_set, train_frame)
            train_predict_time = str2time_stamp(train_file_set[1][0], re=u'date/%Y-%m-%d')
            self.training_feature_extractor = FeatureExtractor(self.training_set, train_predict_time)
        self.test_set = None
        if test_file_set:
            state_listener('Loading Test Set...')
            self.test_set = DataSet(test_file_set[0], test_file_set[1], self.item_set, test_frame, is_train_set=False)
            test_predict_time = str2time_stamp(test_file_set[1][0], re=u'date/%Y-%m-%d')
            self.test_feature_extractor = FeatureExtractor(self.test_set, test_predict_time)

    def extract_feature(self, dir):
        if self.train_file_set:
            state_listener('Extracting Training Set Feature...')
            self.training_feature_extractor.extract_feature(dir + u'/train_features/')
        if self.test_file_set:
            state_listener('Extracting Test Set Feature...')
            self.test_feature_extractor.extract_feature(dir + u'/test_features/')
