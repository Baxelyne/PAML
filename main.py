# -*- coding: utf-8 -*-
# @FileName: main.py

from __future__ import print_function
import argparse
import random
import numpy as np
import os
from engines.PAML import PAML
from engines.DataManager import DataManager
from engines.Configer import Configer
from engines.utils import get_logger


def set_env(configs):
    random.seed(configs.seed)
    np.random.seed(configs.seed)


def fold_check(configs):
    datasets_fold = 'datasets_fold'
    assert hasattr(configs, datasets_fold), "item datasets_fold not configured"

    if not os.path.exists(configs.datasets_fold):
        print("datasets fold not found")
        exit(1)

    checkpoints_dir = 'checkpoints_dir'
    if not os.path.exists(configs.checkpoints_dir) or \
            not hasattr(configs, checkpoints_dir):
        print("checkpoints fold not found, creating...")
        cides = configs.checkpoints_dir.split('/')
        if len(cides) == 2 and os.path.exists(cides[0]) \
                and not os.path.exists(configs.checkpoints_dir):
            os.mkdir(configs.checkpoints_dir)
        else:
            os.mkdir("checkpoints")

    vocabs_dir = 'vocabs_dir'
    if not os.path.exists(configs.vocabs_dir):
        print("vocabs fold not found, creating...")
        if hasattr(configs, vocabs_dir):
            os.mkdir(configs.vocabs_dir)
        else:
            os.mkdir(configs.datasets_fold + "/vocabs")

    log_dir = 'log_dir'
    if not os.path.exists(configs.log_dir):
        print("log fold not found, creating...")
        if hasattr(configs, log_dir):
            os.mkdir(configs.log_dir)
        else:
            os.mkdir(configs.datasets_fold + "/vocabs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning with PAML')
    parser.add_argument('--config_file', default='system.config', help='Configuration File')
    args = parser.parse_args()
    configs = Configer(config_file=args.config_file)

    fold_check(configs)
    logger = get_logger(configs.log_dir)
    configs.show_data_summary(logger)
    set_env(configs)

    mode = configs.mode.lower()

    dataManager = DataManager(configs, logger)
    model = PAML(configs, logger, dataManager)
    if mode == 'train':
        logger.info("mode: train")
        model.train()
    elif mode == 'test':
        logger.info("mode: test")
        model.test()
