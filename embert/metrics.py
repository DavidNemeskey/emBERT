#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics-related code. Not used currently, but could be if early stopping is
required.
"""

from dataclasses import dataclass
from enum import auto, Enum
import logging
from operator import lt, gt
import re

from torch import nn


@dataclass
class Metrics:
    """
    Extracts and stores the main classification metrics from the output of
    :func:`seqeval.metrics.classification_report`.

    .. note::
    Words for seqeval 0.0.5, later versions ouput the results slightly
    differently.
    """
    metric_p = re.compile('avg / total\s+(\S+)\s+(\S+)\s+(\S+)')

    def __init__(self, report):
        m = self.metric_p.search(report)
        self.precision, self.recall, self.f1 = map(float, m.groups())

    precision: float
    recall: float
    f1: float


class EarlyStoppingMode(Enum):
    """Indicates whether the metric should be minimized or maximized."""
    MIN = auto()
    MAX = auto()


class EarlyStopping:
    """
    Early stops the training if the validation metric doesn't improve after a
    given patience.
    
    Based on the code at https://github.com/Bjarten/early-stopping-pytorch,
    adapted to the Huggingface pretrained models and modified to support
    any metric.

    .. note::
    This class can be used without a model. This might be useful for testing
    purposes.
    """
    def __init__(self, patience: int = 7,
                 mode: EarlyStoppingMode = EarlyStoppingMode.MIN,
                 delta: float = 0, output_dir: str = '.'):
        """
        :param patience: how long to wait after last time validation loss
                         improved. Default: 7
        :param mode: whether to minimize (:attr:`EarlyStoppingMode.MIN`) or
                     maximize (:attr:`EarlyStoppingMode.MAX`) the metric.
        :param delta: minimum change in the monitored quantity to qualify as
                      an improvement. Default: 0
        :param output_dir: the directory to which the checkpoint file is
                           written.
        """
        self.patience = patience
        self.counter = 0
        self.output_dir = output_dir

        self.best_score = None
        self.early_stop = False
        if mode is EarlyStoppingMode.MIN:
            self.delta = -delta
            self.op = lt
        else:
            self.delta = delta
            self.op = gt

    def __call__(self, metric: float, model: nn.Module) -> bool:
        """
        Checks if the training should stop early in light of the latest
        validation metric. Sets :attr:`EarlyStopping.early_stop` attribute
        to `True` if it should.

        :returns: :attr:`EarlyStopping.early_stop`.
        """
        if self.best_score is None or self.op(metric, self.best_score + self.delta):
            self.save_checkpoint(metric, model)
            self.counter = 0
        else:
            self.counter += 1
            logging.debug(f'EarlyStopping counter: {self.counter} '
                          f'out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, metric, model):
        """Saves model when the validation metric improves."""
        old_value = (f'{self.best_score:.6f}'
                     if self.best_score is not None else '')
        logging.debug(f'Validation metric improved ({old_value}'
                      f' --> {metric:.6f}). Saving model ...')
        # Only save the model itself, not the parallel wrapper
        if model:
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(self.output_dir)
        # torch.save(model.state_dict(), 'checkpoint.pt')
        self.best_score = metric
