#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


class Logger():

    def __init__(self, logpath, attr_list):
        self.logfile = open(logpath + '.csv', 'w')
        self.logfile.write(','.join(attr_list))


    def add(self, *args):
        self.logfile.write('\n' + ','.join(list(map(lambda x: str(x), args))))


    def close(self):
        self.logfile.close()


def get_available_filepath(*args):
    filepath, exist = os.path.join(*args), True
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        exist = False
    return filepath, exist


