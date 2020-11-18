#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation / prediction functions."""

from typing import Any, Generator, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data_wrapper import DataWrapper
from .viterbi import ReverseViterbi


def log_softmax(
    model: nn.Module, wrapper: DataWrapper
) -> Generator[Any, None, None]:
    """
    Runs the prediction up to the point of applying the log softmax function.
    This allows us to apply the Viterbi algorithm to find the most probable
    tag sequence.

    Yields numpy arrays of ``batch_size x seq_len x num_classes`` dimensions.
    ``seq_len`` will equal to the number of tokens in the sequence, but
    in fact, the model outputs tag probabilities per word, so only the first
    few items are used (up to the ``[SEP]`` token).
    """
    model.eval()
    for (
        input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask
    ) in wrapper:
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None,
                           valid_ids=valid_ids, attention_mask_label=l_mask)[0]
            print('logits1', logits.shape, logits, sep='\n')
        log_probs = F.log_softmax(logits, dim=2)
        # Jumping past [CLS], the first token
        # TODO make it more flexible, e.g. skip all "admin" tokens
        log_probs = log_probs.detach().cpu().numpy()[:, 1:]
        print('log_probs', log_probs.shape, log_probs, np.exp(log_probs),
              np.exp(log_probs).sum(axis=2), sep='\n')
        yield input_ids, log_probs


def predict(
    model: nn.Module, wrapper: DataWrapper, viterbi: ReverseViterbi = None
) -> List[List[str]]:
    """Predicts the labels for all sentences in _wrapper_."""
    sep_id = len(wrapper.get_labels())  # [SEP] is always the last label

    y_pred = []
    model.eval()
    for input_ids, log_probs in log_softmax(model, wrapper):
        # Run ReverseViterbe per batch
        for seq, log_prob in enumerate(log_probs):
            max_prob = np.argmax(log_prob, axis=1)
            seq_len = np.where(max_prob == sep_id)[0][0]
            if viterbi:
                # Note: the TokenClassifier shift label indices up by one,
                #       introducing a new 0 label, which basically means
                #       "untrained" / "undefined". Hence the 1: ... + 1 below.
                state_probs = log_prob[:seq_len, 1:viterbi.num_states() + 1].T
                vprob = [e + 1 for e in viterbi(state_probs, logp=True)]
                max_labels = [wrapper.id_to_label(elem) for elem in vprob]
            else:
                max_labels = [wrapper.id_to_label(elem)
                              for elem in max_prob[:seq_len]]

            y_pred.append(max_labels)

    return y_pred
