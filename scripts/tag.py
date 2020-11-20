#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A simple script that reads files and tags the sentences therein with the
selected model. The model is selected by its alias in emBERT.
"""

import argparse
import logging
import os

from embert.data_format import all_formats, get_format_reader
from embert.embert import EmBERT


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("input_dir",
                        help='The input directory. All files in it are taken '
                             'as input.')
    parser.add_argument("output_dir",
                        help='The output directory. The tagged versions of '
                             'the input files are written here.')
    parser.add_argument("--task-name", required=True,
                        help='The emBERT task. The respective configuration '
                             'file must exist in the configs directory and '
                             'the corresponding model must have been already '
                             'downloaded to the models directory.')
    parser.add_argument('--data_format', required=True, choices=all_formats(),
                        help='The data format of the input files.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    em = EmBERT(task=args.task_name)
    reader = get_format_reader(args.data_format)
    for input_file in os.listdir(args.input_dir):
        with open(os.path.join(args.output_dir, input_file), 'wt') as outf:
            for sentence, labels in reader(os.path.join(args.input_dir, input_file)):
                tagged = em.process_sentence([[w] for w in sentence], [0])
                for token in tagged:
                    print('\t'.join(token), file=outf)
                print(file=outf)


if __name__ == "__main__":
    main()
