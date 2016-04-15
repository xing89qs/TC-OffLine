#!/usr/bin/env python
# -- coding:utf-8 --

import os
from global_var import *


def split_record_by_date():
    record_table_file = open(DIRECTORY + u'tianchi_fresh_comp_train_user1.csv', 'r')
    output_dir = DIRECTORY + u'date/'
    os.mkdir(output_dir)
    record_table_file.readline()
    file_list = {}
    for line in record_table_file:
        record = line.strip()
        date = record.split(',')[5].split(' ')[0]
        if date not in file_list:
            file_list[date] = open(output_dir + date + ".txt", 'w')
        output_file = file_list[date]
        output_file.write(record)
        output_file.write('\n')

    for _file in file_list.values():
        _file.close()


if __name__ == '__main__':
    split_record_by_date()
