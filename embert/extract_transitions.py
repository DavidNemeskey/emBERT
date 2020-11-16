#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extracts the label->label transition matrix for the Viterbi algorithm.
"""

from collections import Counter, defaultdict
import os

import numpy as np

from embert.data_format import get_format_reader
from embert.processors import get_processor
from embert.utils import pairwise


def extract_transitions(train_tsv: str):
    """Extracts the label->label transition matrix."""
    init_stats = Counter()
    transitions = defaultdict(Counter)
    for _, labels in get_format_reader('tsv')(train_tsv):
        init_stats[labels[0]] += 1
        for l1, l2 in pairwise(labels):
            transitions[l1][l2] += 1
    return init_stats, transitions


def extract_transitions2(processor, label_map):
    """Extracts the label->label transition matrix."""
    init_stats = np.zeros(len(label_map), dtype=float)
    transitions = np.zeros((len(label_map), len(label_map)), dtype=float)
    for _, labels in processor.reader(os.path.join(processor.data_dir, 'train.txt')):
        init_stats[label_map[labels[0]]] += 1
        for l1, l2 in pairwise(labels):
            transitions[label_map[l1], label_map[l2]] += 1
    return init_stats, transitions


def main2():
    import sys
    processor_cls = get_processor(sys.argv[2])
    processor = processor_cls(sys.argv[1], get_format_reader('tsv'))
    idx2label = processor.get_labels()
    label2idx = {label: i for i, label in enumerate(idx2label)
                 if not label.startswith('[')}
    init_stats, transitions = extract_transitions2(processor, label2idx)
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


def main():
    import sys
    init_stats, transitions = extract_transitions(sys.argv[1])
    print('INIT:\n')
    init_sum = sum(init_stats.values())
    for k, v in init_stats.most_common():
        print(f'{k}: {v} ({v / init_sum})')

    print('\n\n\nTRANSITIONS:\n')
    for l1, trans in transitions.items():
        tr_sum = sum(trans.values())
        for l2, v in trans.most_common():
            print(f'{l1} -> {l2}: {v} ({v / tr_sum})')


if __name__ == "__main__":
    main2()
