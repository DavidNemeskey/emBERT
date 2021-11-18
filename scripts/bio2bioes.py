#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converts files from BIO to BIOES (more specifically, BIOE1) format.

The version of BIO handled here is the one where 1-* is represented by B-*.
"""

import argparse
import os
import os.path as op


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', help='the input directory')
    parser.add_argument('output_dir', help='the output directory')
    parser.add_argument('--field', '-f', type=int, default=-1,
                        help='the index of the field to be converted (-1).')
    return parser.parse_args()


def read_sentences(input_file: str):
    """Reads _input_file_ (CoNLL format) sentence-by-sentence."""
    with open(input_file) as inf:
        sentence = []
        has_normal_lines = False
        for line_no, line in enumerate(map(str.strip, inf)):
            if not line:
                if sentence:
                    yield sentence
                sentence = []
                has_normal_lines = False
            elif line.startswith('# '):
                if has_normal_lines:
                    raise ValueError(f'Comment after regular line at '
                                     f'{input_file}:{line_no}')
                sentence.append([line])
            else:
                has_normal_lines = True
                sentence.append(line.split('\t'))
        if sentence:
            yield sentence


def convert(infile: str, outfile: str, field_index: int = -1):
    """Does the actual conversion."""
    with open(outfile, 'wt') as outf:
        for sentence in read_sentences(infile):
            # Let's get rid of comments first
            for begin, token in enumerate(sentence):
                if token[0].startswith('# '):
                    print('\t'.join(token), file=outf)
                else:
                    break

            # The conversion
            out_of_group = True
            for i in range(len(sentence) - 1, begin - 1, -1):
                field = sentence[i][field_index]
                if field[0] == 'I':
                    if out_of_group:
                        sentence[i][field_index] = f'E{field[1:]}'
                        out_of_group = False
                elif field[0] == 'B':
                    if out_of_group:
                        sentence[i][field_index] = f'1{field[1:]}'
                    out_of_group = True
                elif field[0] == 'O':
                    out_of_group

            for token in sentence[begin:]:
                print('\t'.join(token), file=outf)
            print(file=outf)


def main():
    args = parse_arguments()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    input_files = sorted(fpath
                         for f in os.listdir(args.input_dir)
                         if op.isfile(fpath := op.join(args.input_dir, f)))
    output_files = sorted(op.join(args.output_dir, op.basename(f))
                          for f in input_files)
    for infile, outfile in zip(input_files, output_files):
        convert(infile, outfile, args.field)


if __name__ == "__main__":
    main()
