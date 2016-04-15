#!/usr/bin/env python
# -- coding:utf-8 --

from pandas import DataFrame, Series
from global_var import *
import math
import numpy as np
import pandas as pd


class Pairs(object):
    def __init__(self):
        self.max_time = 0
        self.count = 0


class UserItemBehaviorPairs(Pairs):
    def __init__(self):
        Pairs.__init__(self)


class UserBehaviorPairs(Pairs):
    def __init__(self):
        Pairs.__init__(self)


class ItemBehaviorPairs(Pairs):
    def __init__(self):
        Pairs.__init__(self)


class UserCategoryBehaviorPairs(Pairs):
    def __init__(self):
        Pairs.__init__(self)


class FeatureExtractor(object):
    def __init__(self, data_set, predict_time):
        self.data_set = data_set
        self.predict_time = predict_time

    def __minus(self, x):
        return self.predict_time - x

    def __extract(self, record_frame, group_keys, prefix):
        # __SPLIT = {86400 * 7: 'day7'}
        # for day, __str in __SPLIT.items():
        # prefix_str = map(lambda x: __str + '_' + x, prefix)
        # new_frame = record_frame[record_frame.time >= self.predict_time - day]
        group = record_frame.groupby(group_keys + ['behavior'], as_index=False)
        frame = group['time'].agg(
            {'count': np.size, 'max_time': np.max})
        group = frame.groupby(['behavior'])
        __BEHAVIORS = [BROWSE, FAVOR, ADD_TO_CART, BUY]
        for _i in xrange(len(__BEHAVIORS)):
            if __BEHAVIORS[_i] in group.groups:
                _frame = group.get_group(__BEHAVIORS[_i])
            else:
                _frame = pd.DataFrame(
                    columns=frame.columns)
            _frame = _frame.rename(
                columns={'count': prefix[_i] + '_count',
                         'max_time': prefix[_i] + '_max_time'},
                copy=False)
            del _frame['behavior']
            self.feature_frame = pd.merge(self.feature_frame, _frame, how='outer', on=group_keys)
            self.feature_frame[prefix[_i] + '_count'].fillna(0, inplace=True,
                                                             downcast='infer')
            self.feature_frame[prefix[_i] + '_max_time'].fillna(self.predict_time,
                                                                inplace=True,
                                                                downcast='infer')
            self.feature_frame[prefix[_i] + '_max_time'] = self.feature_frame[
                prefix[_i] + '_max_time'].apply(
                self.__minus)

    def __has_buy(self, x):
        tmp = min(x['ui_brow_max_time'], x['ui_fav_max_time'], x['ui_atc_max_time'])
        if (tmp < self.predict_time and x['ui_buy_max_time'] < self.predict_time) and tmp <= x['ui_buy_max_time']:
            return False
        return True

    def __get_has_buy_frame(self, record_frame):
        __PREFIX_STR = ['ui_brow', 'ui_fav', 'ui_atc', 'ui_buy']
        user_item_frame = record_frame[['uid', 'iid']].drop_duplicates()
        new_frame = record_frame[record_frame.time >= self.predict_time - 86400]
        group = new_frame.groupby(['uid', 'iid'] + ['behavior'], as_index=False)
        frame = group['time'].agg(
            {'count': np.size, 'max_time': np.max})
        group = frame.groupby(['behavior'])
        __BEHAVIORS = [BROWSE, FAVOR, ADD_TO_CART, BUY]
        for _i in xrange(len(__BEHAVIORS)):
            if __BEHAVIORS[_i] in group.groups:
                _frame = group.get_group(__BEHAVIORS[_i])
            else:
                _frame = pd.DataFrame(
                    columns=frame.columns)
            _frame = _frame.rename(
                columns={'count': __PREFIX_STR[_i] + '_count', 'max_time': __PREFIX_STR[_i] + '_max_time'}, copy=False)
            del _frame['behavior']
            user_item_frame = pd.merge(user_item_frame, _frame, how='outer', on=['uid', 'iid'])
            user_item_frame.fillna(0, inplace=True, downcast='infer')
        has_buy_frame = user_item_frame[user_item_frame.apply(self.__has_buy, axis=1)][['uid', 'iid']]
        self.has_buy_dict = set()
        for _ in has_buy_frame.values:
            self.has_buy_dict.add(tuple(_))

    def __filter(self, x):
        return (x['uid'], x['iid']) not in self.has_buy_dict

    def __calculate(self):
        state_listener('Calculating...')

        record_frame = self.data_set.record_frame

        self.user_item_frame = record_frame[['label', 'uid', 'iid']].drop_duplicates()
        self.item_category_frame = record_frame[['iid', 'category']].drop_duplicates()

        self.feature_frame = pd.merge(self.user_item_frame, self.item_category_frame, how='outer', on='iid')

        user_behavior_group = record_frame.groupby(['uid'], as_index=False)
        self.user_behavior_frame = user_behavior_group['time'].agg({'ub_count': np.size, 'ub_max_time': np.max})

        item_behavior_group = record_frame.groupby(['iid'], as_index=False)
        self.item_behavior_frame = item_behavior_group['time'].agg({'ib_count': np.size, 'ib_max_time': np.max})

        __PREFIX_STR = ['ui_brow', 'ui_fav', 'ui_atc', 'ui_buy']
        self.__extract(record_frame, ['uid', 'iid'], __PREFIX_STR)

        __PREFIX_STR = ['u_brow', 'u_fav', 'u_atc', 'u_buy']
        self.__extract(record_frame, ['uid'], __PREFIX_STR)

        __PREFIX_STR = ['i_brow', 'i_fav', 'i_atc', 'i_buy']
        self.__extract(record_frame, ['iid'], __PREFIX_STR)

        __PREFIX_STR = ['c_brow', 'c_fav', 'c_atc', 'c_buy']
        self.__extract(record_frame, ['category'], __PREFIX_STR)

        __PREFIX_STR = ['uc_brow', 'uc_fav', 'uc_atc', 'uc_buy']
        self.__extract(record_frame, ['uid', 'category'], __PREFIX_STR)

        if not self.data_set.is_train_set:
            self.feature_frame = self.feature_frame[self.feature_frame['iid'].isin(self.data_set.item_set)]
            self.feature_frame.reindex(xrange(len(self.feature_frame)))

        '''
            self.__get_has_buy_frame(record_frame)
            self.feature_frame = self.feature_frame[self.feature_frame.apply(self.__filter, axis=1)]
        '''

    def __to_vec(self, folder):
        for _ in self.feature_frame.columns[4:]:
            self.feature_frame[['uid', 'iid', _]].to_csv(folder + '_' + _ + '.csv')
        self.feature_frame[['uid', 'iid', 'label']].drop_duplicates().to_csv(folder + 'label.csv')

    def extract_feature(self, feature_folder):
        self.__calculate()
        self.__to_vec(feature_folder)
