#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extracts the label->label transition matrix for the Viterbi algorithm.
"""

from typing import List

import numpy as np

# from embert.data_format import get_format_reader
from embert.processors import DataSplit
# from embert.processors import DataSplit, get_processor
from embert.utils import pairwise


def extract_transitions(processor):
    """
    Extracts the label->label transition matrix for the Viterbi algorithm.

    .. note::
    At the moment, this function is not used, because for small training
    corpora, such as Szeged NER, valid sequences might be missing or have
    such a low probability that the algorithm would erroneously prevent them
    from being emitted. Use :func:`default_transitions`, below.
    """
    label_map = {label: i for i, label in enumerate(processor.get_labels())
                 if not label.startswith('[')}
    init_stats = np.zeros(len(label_map), dtype=float)
    transitions = np.zeros((len(label_map), len(label_map)), dtype=float)
    for _, labels in processor.get_file(DataSplit.TRAIN):
        init_stats[label_map[labels[0]]] += 1
        for l1, l2 in pairwise(labels):
            transitions[label_map[l1], label_map[l2]] += 1
    init_norm = init_stats / sum(init_stats)
    trans_norm = transitions / transitions.sum(axis=1)[:, np.newaxis]

    return init_norm, trans_norm


def default_transitions(labels: List[str]):
    """
    Generates default transitions from a BIO (the BII variety) or BIOES/1
    label set. All valid transitions from a state have uniform probabilities,
    and invalid transitions have 0.
    """
    label_map = {label: i for i, label in enumerate(labels)
                 if not label.startswith('[')}
    l_bs = {label for label in labels if label.startswith('B-')}
    l_1s = {label for label in labels
            if label.startswith('1-') or label.startswith('S-')}
    l_is = {label for label in labels if label.startswith('I-')}
    l_es = {label for label in labels if label.startswith('E-')}
    l_o = {'O'}

    l_start = l_bs | l_1s | l_o

    if len(l_bs) != len(l_is):
        raise ValueError('Number of B- and I- labels do not match.')
    if len(l_es) and len(l_es) != len(l_is):
        raise ValueError('Number of I- and E- labels do not match.')

    init_stats = np.zeros(len(label_map), dtype=float)
    for label in l_start:
        init_stats[label_map[label]] = 1

    transitions = np.zeros((len(label_map), len(label_map)), dtype=float)
    for l1 in l_o:
        for l2 in l_start:
            transitions[label_map[l1], label_map[l2]] = 1
    for l1 in l_bs:
        for l2 in l_is | l_es:
            if l1[1:] == l2[1:]:
                transitions[label_map[l1], label_map[l2]] = 1
    for l1 in l_is:
        transitions[label_map[l1], label_map[l1]] = 1
    if l_es:
        for l1 in l_is:
            for l2 in l_es:
                if l1[1:] == l2[1:]:
                    transitions[label_map[l1], label_map[l2]] = 1
        for l1 in l_es:
            for l2 in l_start:
                transitions[label_map[l1], label_map[l2]] = 1
    else:
        for l1 in l_is:
            for l2 in l_start:
                transitions[label_map[l1], label_map[l2]] = 1
    if l_1s:
        for l1 in l_1s:
            for l2 in l_start:
                transitions[label_map[l1], label_map[l2]] = 1

    init_norm = init_stats / init_stats.sum()
    trans_norm = transitions / transitions.sum(axis=1)[:, np.newaxis]
    return init_norm, trans_norm


def save_viterbi(viterbi_file, init_stats, transitions):
    np.savez_compressed(viterbi_file,
                        init_stats=init_stats, transitions=transitions)


def load_viterbi(viterbi_file):
    npz = np.load(viterbi_file)
    return npz['init_stats'], npz['transitions']


# def main():
#     import sys
#     processor_cls = get_processor(sys.argv[2])
#     processor = processor_cls(sys.argv[1], get_format_reader('tsv'))
#     idx2label = processor.get_labels()
#     init_stats, transitions = default_transitions(processor)
#     print('INIT:\n')
#     init_norm = init_stats / sum(init_stats)
#     for idx, v in enumerate(init_stats):
#         if v != 0:
#             print(f'{idx2label[idx]}: {v} ({init_norm[idx]})')
#
#     print('\n\n\nTRANSITIONS:\n')
#     for l1, trans in enumerate(transitions):
#         tr_sum = sum(trans)
#         for l2, v in enumerate(trans):
#             if v != 0:
#                 print(f'{idx2label[l1]} -> {idx2label[l2]}: {v} ({v / tr_sum})')
#
#
# if __name__ == "__main__":
#     main()
