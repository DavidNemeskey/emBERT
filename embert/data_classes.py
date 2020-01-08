#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The basic classes used to represent data in the package."""

from dataclasses import dataclass
from typing import List


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


@dataclass
class InputFeatures:
    """The features of a single training sequence."""
    input_ids: List[int]
    input_mask: List[int]
    segment_ids: List[int]
    label_ids: List[int]
    valid_ids: List[int]
    label_mask: List[int]

    def pad(self, sequence_length):
        """Pads the lists to _sequence_length_."""
        how_many = sequence_length - len(self.input_ids)
        if how_many > 0:
            pad = [0] * how_many
            for field, value in self.__dict__.items():
                value.extend(pad if field != 'valid_ids' else [1] * how_many)
        # label_ids and label_mask are shorter (|words| instead of |tokens|)
        label_pad = [0] * (sequence_length - len(self.label_ids))
        self.label_ids.extend(label_pad)
        self.label_mask.extend(label_pad)
        for field, value in self.__dict__.items():
            assert len(value) == sequence_length, \
                f'len({field}) = {len(value)} != {sequence_length}'
