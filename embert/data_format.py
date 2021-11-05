#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data format readers."""
import logging


def read_tsv(filename):
    """
    Reads a tsv file; extracts the surface word form and the class associated
    with each token. The word form is assumed to be first, the class the last
    column.
    """
    # TODO: A proper CoNLL(-U Plus) reader
    with open(filename) as inf:
        data = []
        sentence, labels = [], []
        for line in map(str.strip, inf):
            if not line:
                if sentence:
                    data.append((sentence, labels))
                sentence, labels = [], []
            elif not line.startswith('# '):
                fields = line.split('\t')
                sentence.append(fields[0])
                labels.append(fields[-1])
            else:
                logging.info(f'Dropping line {line}...')
        if sentence:
            data.append((sentence, labels))
    return data


_readers = {'tsv': read_tsv}


def all_formats():
    """Returns the list of data format readers available."""
    return _readers.keys()


def get_format_reader(format):
    """Returns the data format reader associated with _format_."""
    return _readers[format]
