#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation / prediction functions."""

from typing import Any, Callable, Generator, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data_wrapper import DataWrapper
from .viterbi import ReverseViterbi


class PredictionResult:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.y_pred = []

    def add_data(self, max_prob_seq, *args):
        max_labels = [self.wrapper.id_to_label(elem)
                      for elem in max_prob_seq]
        self.y_pred.append(max_labels)


class EvaluationResult(PredictionResult):
    def __init__(self, wrapper):
        super().__init__(wrapper)
        self.y_true = []
        self.num_zero_labels = 0
        self.loss = 0
        self.steps = 0

    def add_data(self, max_prob_seq, label_ids, loss):
        super().add_data(max_prob_seq)
        self.num_zero_labels += np.count_nonzero(max_prob_seq == 0)
        self.y_true.append([self.wrapper.id_to_label(elem)
                            for elem in label_ids[:len(max_prob_seq)]])
        self.loss += loss
        self.steps += len(max_prob_seq)

    def get_loss(self):
        return self.loss / self.steps


class Evaluator:
    def __init__(self, model: nn.Module, wrapper: DataWrapper,
                 viterbi: ReverseViterbi = None, it_wrapper: Callable = None):
        """
        Creates a new :class:`Evaluator` object.

        :param model: the model to use for evaluation.
        :param wrapper: the data wrapper.
        :param viterbi: the object implementing the Viterbi algorithm (if used).
        :param it_wrapper: a callable that wraps the data stream generated by
                           _wrapper_. An example would be ``tqdm``. The default
                           is ``None``.
        """
        self.model = model
        self.wrapper = wrapper
        self.viterbi = viterbi
        if it_wrapper:
            self.it_wrapper = it_wrapper
        else:
            self.it_wrapper = lambda it: it
        # [SEP] is always the last label
        self.sep_id = len(self.wrapper.get_labels())

    def predict(self):
        """Predicts the labels for all sentences in :attr:`wrapper`."""
        return self(True)

    def evaluate(self):
        """
        Same as predict, but all kinds of additional information are
        returned, such as the loss, the number of zero label ids and the gold
        label sequence.
        """
        return self(False)

    def __call__(self, predict_only: bool = False) -> List[List[str]]:
        """
        Predicts the labels for all sentences in :attr:`wrapper`.
        If _predict_only_ is ``False``, all kinds of additional information are
        returned, such as the loss, the number of zero label ids and the gold
        label sequence.
        """
        self.model.eval()
        result = (PredictionResult if predict_only
                  else EvaluationResult)(self.wrapper)
        for log_probs, label_ids, loss in self.log_softmax(not predict_only):
            # Run ReverseViterbi per batch
            for seq, log_prob in enumerate(log_probs):
                max_prob = np.argmax(log_prob, axis=1)
                seq_len = np.where(max_prob == self.sep_id)[0][0]
                if self.viterbi:
                    # Note: the TokenClassifier shift label indices up by one,
                    #       introducing a new 0 label, which basically means
                    #       "untrained" / "undefined". Hence the 1: ... + 1 below.
                    state_probs = log_prob[:seq_len,
                                           1:self.viterbi.num_states() + 1].T
                    max_prob_seq = [e + 1 for e in self.viterbi(state_probs, logp=True)]
                else:
                    max_prob_seq = max_prob[:seq_len]
                labels = label_ids[seq][:seq_len]
                assert labels.size()[0] == len(max_prob_seq), \
                    f'{labels.size()[0]} != {len(max_prob_seq)}'

                result.add_data(max_prob_seq, labels,
                                loss[0].item() if loss and seq == 0 else 0)

        return result

    def log_softmax(self, with_label_ids: bool = False) -> Generator[Any, None, None]:
        """
        Runs the prediction up to the point of applying the log softmax
        function. This allows us to apply the Viterbi algorithm to find the
        most probable tag sequence.

        Yields numpy arrays of ``batch_size x seq_len x num_classes``
        dimensions. ``seq_len`` will equal to the number of tokens in the
        sequence, but in fact, the model outputs tag probabilities per word,
        so only the first few items are used (up to the ``[SEP]`` token).
        """
        self.model.eval()
        for (
            input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask
        ) in self.it_wrapper(self.wrapper):
            gold_labels = label_ids if with_label_ids else None

            with torch.no_grad():
                *loss, logits = self.model(
                    input_ids, segment_ids, input_mask, labels=gold_labels,
                    valid_ids=valid_ids, attention_mask_label=l_mask
                )
            log_probs = F.log_softmax(logits, dim=2)
            # Jumping past [CLS], the first token
            # TODO make it more flexible, e.g. skip all "admin" tokens
            log_probs = log_probs.detach().cpu().numpy()[:, 1:]
            yield log_probs, label_ids[:, 1:], loss
