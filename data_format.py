#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data format readers."""


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
            if line:
                fields = line.split('\t')
                sentence.append(fields[0])
                labels.append(fields[-1])
            else:
                if sentence:
                    data.append((sentence, labels))
                sentence, labels = [], []
        if sentence:
            data.append((sentence, labels))
    return data


_readers = {'stupid': read_stupid, 'tsv': read_tsv}
