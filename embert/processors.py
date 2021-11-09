#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data processors collect the data files for the train / devel /test splits
and the labels used in the train corpus.
"""

from enum import Enum
from itertools import chain
import logging
import os
import random
import re
from typing import Callable, Iterable, List, Tuple

from .data_classes import InputExample


class DataSplit(Enum):
    """Enum for the train / devel / test split."""
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class DataProcessor:
    """
    Data processors collect the data files for the train / devel /test splits
    and the labels used in the train corpus. Files are identified in two ways:
    - if there is a file named after a data split (e.g. ``train.txt``) right
      under the data directory, that file is added to the list
    - if there is a directory named after the split (e.g. ``train``), all
      files with extensions ``.txt``, ``.tsv`` and ``.conll(up)`` are added
    - if a per-split data directory is specified (e.g. ``train_dir``), all
      files with extensions ``.txt``, ``.tsv`` and ``.conll(up)`` are added.
    """
    ALIASES = {DataSplit.VALID: {'devel'}}
    EXT_PATTERN = re.compile('[.](?:tsv|txt|conll(?:up)?)$')

    def __init__(self, data_dir: str,
                 format_reader: Callable[[str], Tuple[List[str], List[str]]],
                 **kwargs: str):
        self.data_dir = data_dir
        self.reader = format_reader
        self.split_dirs = {split: kwargs.get(f'{split.value}_dir')
                           for split in DataSplit}
        self.labels = self.read_labels() + ['[CLS]', '[SEP]']
        logging.info(f'Labels are: {self.labels}.')

    def read_labels(self):
        """
        Reads the labels from all splits of the training corpus and saves it
        to a file called _labels.embert_ in the data directory. If the file
        already exists, it reads the labels from there instead of the corpus.
        """
        labels_file = os.path.join(self.data_dir, 'labels.embert')
        if os.path.isfile(labels_file):
            with open(labels_file) as inf:
                return [label for line in inf.read().split('\n')
                        if (label := line.strip())]
        else:
            labels = set()
            for split in DataSplit:
                for sentence, sent_labels in self.get_split_data(split):
                    labels.update(sent_labels)
            sorted_labels = ['O'] + sorted(labels - {'O'})
            with open(labels_file, 'wt') as outf:
                outf.write('\n'.join(sorted_labels))
            return sorted_labels

    @staticmethod
    def all_files_from_directory(directory: str) -> List[str]:
        """Enumerates all data files in _dir_ recursively."""
        return [
            os.path.join(sub_dir, sub_file)
            for sub_dir, _, sub_files in os.walk(directory)
            for sub_file in sub_files
            if DataProcessor.EXT_PATTERN.search(sub_file)
        ]

    def get_files(self, split: DataSplit) -> List[str]:
        """Gets all data files associated with _split_."""
        split_aliases = {split.value} | DataProcessor.ALIASES.get(split, set())
        split_pattern = re.compile(
            f'(?:{"|".join(split_aliases)})[.](?:tsv|txt|conll(?:up)?)'
        )
        if (split_dir := self.split_dirs.get(split)):
            split_files = DataProcessor.all_files_from_directory(split_dir)
        else:
            split_files = []
            for file_name in sorted(os.listdir(self.data_dir)):
                file_path = os.path.join(self.data_dir, file_name)
                if file_name in split_aliases and os.path.isdir(file_path):
                    split_files.extend(
                        DataProcessor.all_files_from_directory(file_path))
                elif split_pattern.fullmatch(file_name):
                    split_files.append(file_path)
        # Shuffling it so that we don't create "blocks" of similar texts in
        # the train stream
        logging.info(f'Files for {split} before shuffling: {split_files}.')
        random.shuffle(split_files)
        return split_files

    def get_split_data(
        self, split: DataSplit
    ) -> Iterable[Tuple[List[str], List[str]]]:
        """
        Returns an iterable of input - output (words and labels in a sentence)
        tuples.
        """
        return chain.from_iterable(map(self.reader, self.get_files(split)))

    def get_examples(self, split: DataSplit) -> List[InputExample]:
        """
        Gets a collection of `InputExample`s for the selected split.
        This is the generic version for the per-split methods below.
        """
        return self.create_examples(self.get_split_data(split), split.value)

    def get_train_examples(self) -> List[InputExample]:
        """
        Gets a collection of `InputExample`s for the train set.
        """
        return self.get_examples(DataSplit.TRAIN)

    def get_dev_examples(self) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the dev set."""
        return self.get_examples(DataSplit.VALID)

    def get_test_examples(self) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the test set."""
        return self.get_examples(DataSplit.TEST)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.labels

    def create_examples(self, lines, set_type):
        """
        Creates :class:`InputExample`s from the content of the data file.
        """
        examples = []
        lines = list(lines)
        for i, (sentence, labels) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            labels = labels
            examples.append(InputExample(guid, text_a, labels=labels))
        return examples
