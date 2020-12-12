import json
import logging
import math
import os
import random

import numpy as np
import torch
from torch import nn, Tensor


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0.001, patience=5, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False
        if math.isnan(metrics):
            return True
        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            return True
        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('illegal mode: ', mode)
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - \
                    (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + \
                    (best * min_delta / 100)


def translate(vocab, pred_tags):
    """ translate id to str """
    if isinstance(pred_tags, Tensor):
        pred_tags = pred_tags.squeeze().tolist()
        if isinstance(pred_tags, int):
            pred_tags = [pred_tags]
    rtn_pred = []
    for t in pred_tags:
        try:
            rtn_pred.append(vocab[t])
        except KeyError:
            rtn_pred.append(vocab[0])

    return rtn_pred


def save_checkpoint(experiment_time, model, optimizer):
    mkdir('./results/checkpoints')
    checkpoint_path = './results/checkpoints/' + experiment_time + '.pth'
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(latest, file_name=None):
    """ load the latest checkpoint """
    checkpoints_dir = './results/checkpoints'

    if latest:
        file_list = os.listdir(checkpoints_dir)
        file_list.sort(key=lambda fn: os.path.getmtime(
            checkpoints_dir + '/' + fn))
        checkpoint = torch.load(checkpoints_dir + '/' + file_list[-1])
        return checkpoint, str(file_list[-1])
    else:
        if file_name is None:
            raise ValueError('checkpoint_path cannot be empty!')
        checkpoint = torch.load(checkpoints_dir + '/' + file_name)
        return checkpoint, file_name


def count_parameters(net: torch.nn.Module):
    """ count the numbers of parameters in the model """
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def weights_init(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0.0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def fix_seed(seed):
    """ fix seed to ensure reproducibility """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(" %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def record_time(start_time, end_time):
    """ get minute & second-level measurement of the asc-time """
    elapsed_time = end_time - start_time
    elapsed_min = int(elapsed_time / 60)
    elapsed_sec = int(elapsed_time - (elapsed_min * 60))
    return elapsed_min, elapsed_sec


def mkdir(dir_path):
    """ create folder if not exists. """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def load_file(path):
    with open(path, 'r') as f:
        return json.load(f)
