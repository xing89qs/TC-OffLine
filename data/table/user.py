#!/usr/bin/env python
# -- coding:utf-8 --


class User(object):

    def __init__(self, uid, geohash):
        self.uid = uid
        self.geohash = geohash
        self.record_list = [[] for _ in xrange(4)]

    def add_record(self, behavior, record):
        self.record_list[behavior-1].append(record)


class Record(object):

    def __init__(self, iid, category, time, line=''):
        self.iid = iid
        self.category = category
        self.time = time
        self.line = line
