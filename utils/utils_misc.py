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


def get_logpath(args):
    if args.multi_loss == 'ce':
        logdir = args.model +'-'+ args.multi_loss
    else:
        logdir = args.model +'-'+ args.multi_loss +'-'+ args.binary_loss

    if args.unbiased:
        LOGDIR = './logs/unbiased'
    else:
        LOGDIR = './logs/denoise'

    filename = 'K'+ str(args.num_classes) +'_N'+ str(args.num_candidates) +'_n'+ str(args.num_data) +'_trial'
    logpath, _ = get_available_filepath(LOGDIR, args.dset_name, logdir, filename)
    return logpath