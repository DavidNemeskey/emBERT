#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Splits a dataset (a dictionary of tsv files) into a train-valid-test splits.
"""

import argparse
from contextlib import ExitStack
import os
import os.path
import random


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', help='the input directory')
    parser.add_argument('output_dir', help='the output directory')
    parser.add_argument('--valid-ratio', '-v', type=float, default=0.1,
                        help='the ratio of the valid split (0-1) [0.1].')
    parser.add_argument('--test-ratio', '-t', type=float, default=0.1,
                        help='the ratio of the test split (0-1) [0.1].')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='the random seed [42].')
    args = parser.parse_args()

    if args.valid_ratio < 0 or args.test_ratio < 0:
        parser.error('--valid-ratio and --test-ratio must be at least 0.')
    if args.valid_ratio + args.test_ratio >= 1:
        parser.error('Valid and test sets should sum to less than the '
                     'full corpus.')
    return args


def read_tsv(file_name):
    sentence = []
    with open(file_name) as inf:
        for line in map(str.strip, inf):
            if line:
                sentence.append(line)
            else:
                if sentence:
                    yield sentence
                sentence = []


def main():
    args = parse_arguments()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    random.seed(args.seed)
    weights = [1 - args.valid_ratio - args.test_ratio,
               args.valid_ratio, args.test_ratio]

    input_files = sorted(os.path.join(args.input_dir, f)
                         for f in os.listdir(args.input_dir))
    files = [open(os.path.join(args.output_dir, 'train.txt'), 'wt'),
             open(os.path.join(args.output_dir, 'valid.txt'), 'wt'),
             open(os.path.join(args.output_dir, 'test.txt'), 'wt')]
    with ExitStack() as stack:
        for f in files:
            stack.enter_context(f)

        for input_file in input_files:
            for sentence in read_tsv(input_file):
                output_file = random.choices(files, weights=weights, k=1)[0]
                print('\n'.join(sentence) + '\n', file=output_file)


if __name__ == "__main__":
    main()
