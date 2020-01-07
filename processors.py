#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data processors."""

from dataclasses import dataclass
from enum import Enum
import os
from typing import Callable, List, Tuple


@dataclass
class InputExample:
    """A single training/test example for simple sequence classification."""
    # Unique id for the example.
    guid: str
    # The untokenized text of the first sequence. For single
    # sequence tasks, only this sequence must be specified.
    text_a: List[str]
    # The untokenized text of the second sequence.
    # Only must be specified for sequence pair tasks.
    text_b: List[str] = None
    # The label of the example. This should be
    # specified for train and dev examples, but not for test examples.
    labels: List[str] = None


class DataSplit(Enum):
    """Enum for the train / devel / test split."""
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir: str,
                 format_reader: Callable[[str], Tuple[List[str], List[str]]]):
        self.data_dir = data_dir
        self.reader = format_reader

    def get_examples(self, split: DataSplit) -> List[InputExample]:
        """
        Gets a collection of `InputExample`s for the selected split.
        This is the generic version for the per-split methods below.
        """
        return self.create_examples(
            self.reader(os.path.join(self.data_dir, f'{split.value}.txt')),
            split.value
        )

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
        raise NotImplementedError()

    def create_examples(self, lines, set_type):
        """
        Creates :class:`InputExample`s from the content of the data file.
        """
        examples = []
        for i, (sentence, labels) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            labels = labels
            examples.append(InputExample(guid, text_a, labels=labels))
        return examples


class CoNLLNerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG",
                "B-LOC", "I-LOC", "[CLS]", "[SEP]"]


class SzegedNerProcessor(DataProcessor):
    def get_labels(self):
        return ['1-LOC', '1-MISC', '1-ORG', '1-PER', 'B-LOC', 'B-MISC',
                'B-ORG', 'B-PER', 'E-LOC', 'E-MISC', 'E-ORG', 'E-PER', 'I-LOC',
                'I-MISC', 'I-ORG', 'I-PER', 'O', '[CLS]', '[SEP]']


class SzegedChunkProcessor(DataProcessor):
    def get_labels(self):
        return ['B-NP', 'I-NP', 'O', '[CLS]', '[SEP]']


class SzegedBIOESChunkProcessor(DataProcessor):
    def get_labels(self):
        return ['1-NP', 'B-NP', 'E-NP', 'I-NP', 'O', '[CLS]', '[SEP]']


# TODO: to IOB / BIOES + Task
_processors = {'conll_ner': CoNLLNerProcessor,
               'szeged_ner': SzegedNerProcessor,
               'szeged_chunk': SzegedChunkProcessor,
               'szeged_bioes_chunk': SzegedBIOESChunkProcessor}


def all_processors():
    """Returns the list of processors available."""
    return _processors.keys()


def get_processor(task_name):
    """Returns the processor associated with _task_name_."""
    return _processors[task_name]