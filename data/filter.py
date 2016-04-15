#!/usr/bin/env python
# -- coding:utf-8 --

from pandas import DataFrame
from table import *
from global_var import *


class Filter(object):
    __MIN_BUY_TIMES = 3
    __MAX_RATE = 1000.0

    def __init__(self):
        pass

    @classmethod
    def is_user_valid(cls, user):
        buy_times = len(user.record_list[BUY])
        browse_times = len(user.record_list[BROWSE])
        add_to_cart_times = len(user.record_list[ADD_TO_CART])
        favor_times = len(user.record_list[FAVOR])
        record_times = buy_times + browse_times + favor_times + add_to_cart_times

        # 过滤规则1
        if record_times == 0:  # 没看过
            return False

        # 过滤规则2

        if buy_times != 0 and float(browse_times) / buy_times > Filter.__MAX_RATE:  # 看的太多买的太少
            return False

        return True


def read_item():
    item_table_file = open(DIRECTORY + u'tianchi_fresh_comp_train_item.csv', 'r')
    item_list = []
    item_table_file.readline()  # 第一行不要
    for line in item_table_file.readlines():
        item = [_ for _ in line.strip().split(',')]
        item[0] = int(item[0])
        item[2] = int(item[1])
        item_list.append(item)
    item_table_file.close()
    return DataFrame(item_list, columns=['iid', 'geohash', 'category'])


def write_user_to_file(file, user):
    for _i in xrange(4):
        for record in user.record_list[_i]:
            file.write(record.line)


def read_record():
    record_table_file = open(DIRECTORY + u'tianchi_fresh_comp_train_user.csv', 'r')
    new_table_file = open(DIRECTORY + u'tianchi_fresh_comp_train_user1.csv', 'w')
    record_table_file.readline()  # 第一行不要
    last_uid = -1
    user = User(-1, ' ')
    for line in record_table_file:
        record = [_ for _ in line.strip().split(',')]
        uid = int(record[0])
        if uid != last_uid:
            if Filter.is_user_valid(user):
                write_user_to_file(new_table_file, user)
            user = User(uid, record[3])
        user.add_record(int(record[2]), Record(record[1], record[4], record[5], line))
        last_uid = uid
    record_table_file.close()
    new_table_file.close()


if __name__ == '__main__':
    #  item_frame = read_item()
    read_record()
