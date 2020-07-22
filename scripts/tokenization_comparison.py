#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compares the tokenization of a BERT model and a SentencePiece vocab file.

Reproduces the vocabulary comparison in the paper.
"""

import argparse
from collections import Counter
import os

from transformers import BertTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', '-i', required=True,
                        help='A directory of tsv files. The files are '
                             'tokenized according to both vocabularies and '
                             'various statistics are collected.')
    parser.add_argument('--model-dir', '-m', required=True,
                        help='The BERT model directory that contains the '
                             'vocabulary (file) of the model.')
    parser.add_argument('--vocab-file', '-v', required=True,
                        help='The standalone vocabulary file (in BERT '
                             'vocabulary format) to compare the model '
                             'tokenizer with.')
    return parser.parse_args()


def collect_words(input_dir):
    c = Counter()
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    for input_file in input_files:
        with open(input_file) as inf:
            for line in map(str.strip, inf):
                if line:
                    c[line.split('\t')[0]] += 1
    return c


def count_wordpieces(words, tokenizer):
    lower = {'sum': 0, 'num': 0, 'type_sum': 0, 'type_num': 0}
    capital = {'sum': 0, 'num': 0, 'type_sum': 0, 'type_num': 0}
    for word, freq in words.items():
        stats = lower if word[0].islower() else capital
        wps = len(tokenizer.tokenize(word))
        stats['type_sum'] += wps
        stats['type_num'] += 1
        stats['sum'] += wps * freq
        stats['num'] += freq
    return {
        'mean': (lower['sum'] + capital['sum']) / (lower['num'] + capital['num']),
        'type_mean': (lower['type_sum'] + capital['type_sum']) / (lower['type_num'] + capital['type_num']),
        'lower_mean': lower['sum'] / lower['num'],
        'lower_type_mean': lower['type_sum'] / lower['type_num'],
        'capital_mean': capital['sum'] / capital['num'],
        'capital_type_mean': capital['type_sum'] / capital['type_num'],
    }


def main():
    args = parse_arguments()

    words = collect_words(args.input_dir)

    mlbt = BertTokenizer(args.model_dir, do_lower_case=False)
    hubt = BertTokenizer(args.vocab_file, do_lower_case=False)

    mlstats = count_wordpieces(words, mlbt)
    hustats = count_wordpieces(words, hubt)

    print(f'Multilingual: {mlstats}')
    print(f'Hungarian: {hustats}')


if __name__ == "__main__":
    main()
