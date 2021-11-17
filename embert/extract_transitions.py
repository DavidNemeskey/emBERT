#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extracts the label->label transition matrix for the Viterbi algorithm.
"""

from typing import Dict, List, Set

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


def bio_transitions(label_map: Dict[str, int], l_bs: Set[str],
                    l_is: Set[str], l_o: Set[str]):
    """
    Generates default transitions from a BIO (the BII variety) label set.
    All valid transitions from a state have uniform probabilities,
    and invalid transitions have 0.
    """
    l_start = l_bs | l_o

    transitions = np.zeros((len(label_map), len(label_map)), dtype=float)
    # In BIO, all states can transition to one of the starting states
    for l1 in label_map.keys():
        for l2 in l_start:
            transitions[label_map[l1], label_map[l2]] = 1
    # B- and I- to I-
    for l1 in l_bs | l_is:
        for l2 in l_is:
            if l1[1:] == l2[1:]:
                transitions[label_map[l1], label_map[l2]] = 1

    return transitions


def bioes_transitions(label_map: Dict[str, int], l_bs: Set[str],
                      l_is: Set[str], l_o: Set[str],
                      l_es: Set[str], l_1s: Set[str]):
    """
    Generates default transitions from BIOES/1 label set.
    All valid transitions from a state have uniform probabilities,
    and invalid transitions have 0.
    """
    l_start = l_bs | l_1s | l_o

    if len(l_bs) != len(l_is) != len(l_es):
        raise ValueError('Number of B-, I- and E- labels do not match.')

    transitions = np.zeros((len(label_map), len(label_map)), dtype=float)
    # O -> O, B-, 1-
    for l1 in l_o | l_1s | l_es:
        for l2 in l_start:
            transitions[label_map[l1], label_map[l2]] = 1
    # B- -> I-, E-
    for l1 in l_bs:
        for l2 in l_is | l_es:
            if l1[1:] == l2[1:]:
                transitions[label_map[l1], label_map[l2]] = 1
    # I- -> I-
    for l1 in l_is:
        transitions[label_map[l1], label_map[l1]] = 1
    # I- -> E-
    for l1 in l_is:
        for l2 in l_es:
            if l1[1:] == l2[1:]:
                transitions[label_map[l1], label_map[l2]] = 1

    return transitions


def default_transitions(labels: List[str]):
    """
    Generates default transitions from BIOES/1 label set.
    All valid transitions from a state have uniform probabilities,
    and invalid transitions have 0.
    """
    # Filtering BERT (model)-specific tags (e.g. [CLS])
    label_map = {label: i for i, label in enumerate(labels)
                 if not label.startswith('[')}
    l_bs = {label for label in labels if label.startswith('B-')}
    l_1s = {label for label in labels
            if label.startswith('1-') or label.startswith('S-')}
    l_is = {label for label in labels if label.startswith('I-')}
    l_es = {label for label in labels if label.startswith('E-')}
    l_o = {'O'}

    l_start = l_bs | l_1s | l_o

    init_stats = np.zeros(len(label_map), dtype=float)
    for label in l_start:
        init_stats[label_map[label]] = 1

    if l_1s:
        transitions = bioes_transitions(label_map, l_bs, l_is, l_o, l_es, l_1s)
    else:
        transitions = bio_transitions(label_map, l_bs, l_is, l_o)

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
