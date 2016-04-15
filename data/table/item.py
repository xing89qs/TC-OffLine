#!/usr/bin/env python
# -- coding:utf-8 --


class Item(object):

    def __init__(self, iid, geohash, category):
        self.iid = iid
        self.geohash = geohash
        self.category = category
