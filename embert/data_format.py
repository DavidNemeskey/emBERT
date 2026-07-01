#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data format readers."""
import logging
from datasets import load_dataset
import pandas as pd
import ast

def read_tsv(filename):
    """
    Reads a tsv file; extracts the surface word form and the class associated
    with each token. The word form is assumed to be first, the class the last
    column.
    """
    # TODO: A proper CoNLL(-U Plus) reader
    with open(filename, encoding='utf-8') as inf:
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
                logging.debug(f'Dropping line {line}...')
        if sentence:
            data.append((sentence, labels))
    return data


def read_csv(filename):
    """
    Reads the ficsort/SzegedNER dataset from Hugging Face and formats it 
    exactly like the old read_tsv function.
    
    Returns:
        A list of tuples containing (sentence_tokens, label_strings).
    """

    print(f"Reading CSV file: {filename}...")

    dataset = pd.read_csv(filename, sep=",")

    dataset['tokens'] = dataset['tokens'].apply(ast.literal_eval)
    dataset['ner'] = dataset['ner'].apply(ast.literal_eval)

    print(f"Dataset columns: {dataset.columns.tolist()}")
    
    data = []
    for _, row in dataset.iterrows():
        sentence = row['tokens']
        # Convert Hugging Face integer tags back to their string representations
        labels = [tag for tag in row['ner']]
        
        data.append((sentence, labels))

    print(data[0][0])

    print(f"Read {len(data)} examples from {filename}.")
        
    return data



_readers = {'tsv': read_tsv, 'csv': read_csv}


def all_formats():
    """Returns the list of data format readers available."""
    return _readers.keys()


def get_format_reader(format):
    """Returns the data format reader associated with _format_."""
    return _readers[format]
