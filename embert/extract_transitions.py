#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extracts the label->label transition matrix for the Viterbi algorithm.
"""

import numpy as np

from embert.data_format import get_format_reader
from embert.processors import DataSplit, get_processor
from embert.utils import pairwise


def extract_transitions(processor):
    """Extracts the label->label transition matrix."""
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


def save_viterbi(viterbi_file, init_stats, transitions):
    np.savez_compressed(viterbi_file,
                        init_stats=init_stats, transitions=transitions)


def load_viterbi(viterbi_file):
    npz = np.load(viterbi_file)
    return npz['init_stats'], npz['transitions']


def main():
    import sys
    processor_cls = get_processor(sys.argv[2])
    processor = processor_cls(sys.argv[1], get_format_reader('tsv'))
    idx2label = processor.get_labels()
    init_stats, transitions = extract_transitions(processor)
    print('INIT:\n')
    init_norm = init_stats / sum(init_stats)
    for idx, v in enumerate(init_stats):
        if v != 0:
            print(f'{idx2label[idx]}: {v} ({init_norm[idx]})')

    print('\n\n\nTRANSITIONS:\n')
    for l1, trans in enumerate(transitions):
        tr_sum = sum(trans)
        for l2, v in enumerate(trans):
            if v != 0:
                print(f'{idx2label[l1]} -> {idx2label[l2]}: {v} ({v / tr_sum})')


if __name__ == "__main__":
    main()
