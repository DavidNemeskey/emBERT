#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Defines the :class:`DataWrapper` class."""

import logging
from typing import List, Type

import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset
from transformers import PreTrainedTokenizer

from embert.data_classes import InputExample, InputFeatures
from embert.processors import DataProcessor, DataSplit


class DataWrapper:
    def __init__(self, processor: DataProcessor, split: DataSplit,
                 sampler_cls: Type[Sampler], batch_size: int,
                 max_seq_length: int, tokenizer: PreTrainedTokenizer,
                 device: torch.device):
        self.processor = processor
        self.split = split
        self.tokenizer = tokenizer
        self.device = device

        self.label_map = {label: i for i, label in
                          enumerate(self.processor.get_labels(), 1)}

        examples = processor.get_examples(split)
        features = self.convert_examples_to_features(
            examples, max_seq_length)

        # TODO: is it possible to do this better?
        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long, device=device)
        all_input_mask = torch.tensor([f.input_mask for f in features],
                                      dtype=torch.long, device=device)
        all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                       dtype=torch.long, device=device)
        all_label_ids = torch.tensor([f.label_ids for f in features],
                                     dtype=torch.long, device=device)
        all_valid_ids = torch.tensor([f.valid_ids for f in features],
                                     dtype=torch.long, device=device)
        all_lmask_ids = torch.tensor([f.label_mask for f in features],
                                     dtype=torch.long, device=device)

        self.data = TensorDataset(all_input_ids, all_input_mask,
                                  all_segment_ids, all_label_ids,
                                  all_valid_ids, all_lmask_ids)
        sampler = sampler_cls(self.data)
        self.dataloader = DataLoader(self.data, sampler=sampler,
                                     batch_size=batch_size)

        self.num_examples = len(self.data)
        self.batch_size = batch_size
        self.num_steps = len(self)

    def convert_examples_to_features(self, examples: List[InputExample],
                                     max_seq_length: int):
        """Loads a data file into a list of :class:`InputFeatures`s."""
        features = []
        for ex_index, example in enumerate(examples):
            tokens = []
            labels = example.labels[:]
            valid = []
            for i, word in enumerate(example.text_a):
                word_tokens = self.tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                for m in range(len(word_tokens)):
                    valid.append(1 if m == 0 else 0)

            # TODO WTF check this out in the original BERT code -- tokens and labels
            # have different lengths, so this doesn't make any sense
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]

            ntokens = ['[CLS]'] + tokens + ['[SEP]']
            segment_ids = [0] * len(ntokens)
            label_ids = [self.label_map[l] for l in ['[CLS]'] + labels + ['[SEP]']]
            label_mask = [1] * len(label_ids)
            valid.insert(0, 1)
            valid.append(1)
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)

            if ex_index < 5:
                logging.info('*** Example ***')
                logging.info(f'guid: {example.guid}')
                logging.info(f'tokens ({len(tokens)}): {" ".join(tokens)}')
                logging.info(f'input_ids ({len(input_ids)}): {" ".join(str(x) for x in input_ids)}')
                logging.info(f'input_mask ({len(input_mask)}): {" ".join(str(x) for x in input_mask)}')
                logging.info(f'segment_ids ({len(segment_ids)}): {" ".join(str(x) for x in segment_ids)}')
                logging.info(f'label_ids ({len(label_ids)}): {" ".join(str(x) for x in label_ids)}')
                logging.info(f'valid_ids ({len(valid)}): {" ".join(str(x) for x in valid)}')
                logging.info(f'label_mask ({len(label_mask)}): {" ".join(str(x) for x in label_mask)}')
                # logging.info("label: %s (id = %d)" % (example.labels, label_ids))

            feature = InputFeatures(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    label_ids=label_ids,
                                    valid_ids=valid,
                                    label_mask=label_mask)
            feature.pad(max_seq_length)
            features.append(feature)
        return features

    def get_labels(self):
        """Returns the label list used in the wrapped dataset."""
        return self.processor.get_labels()

    def get_label_map(self):
        """Returns the label map used in the wrapped dataset."""
        return self.label_map

    def __iter__(self):
        for batch in self.dataloader:
            # Crop tensors to actual length
            actual_seq_length = batch[1].sum(dim=1).max()
            yield (t[:, :actual_seq_length].contiguous() for t in batch)
        # yield from self.dataloader

    def __len__(self) -> int:
        """
        Returns the number of batches in the dataset.

        .. warning::
            Might not work for iterable datasets.
        """
        return len(self.dataloader)


class SentenceWrapper:
    """
    A simple reimplementation of :class:`DataWrapper`. Converts single sentences
    to :class:`InputFeatures` one-by-one; to be used in the emtsv wrapper.
    """
    def __init__(self, processor: DataProcessor, max_seq_length: int,
                 tokenizer: PreTrainedTokenizer, device: torch.device):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = None

    def set_sentence(self, sentence: List[str]):
        """
        Sets _sentence_ as the current sentence to be returned by
        :meth:`__iter__`. Also returns the corresponding :class:`TensorDataset`
        so that it can be used right away, if needed.
        """
        tokens = []
        valid = []
        for i, word in enumerate(sentence):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            for m in range(len(word_tokens)):
                valid.append(1 if m == 0 else 0)

        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[0:(self.max_seq_length - 2)]
            valid = valid[0:(self.max_seq_length - 2)]

        ntokens = ['[CLS]'] + tokens + ['[SEP]']
        segment_ids = [0] * len(ntokens)
        valid.insert(0, 1)
        valid.append(1)
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        labels = torch.empty([1, min(len(sentence), self.max_seq_length)],
                             dtype=torch.long, device=self.device)

        self.dataset = (  # TensorDataset(
            torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0),
            # TODO don't need
            torch.tensor(input_mask, dtype=torch.long, device=self.device).unsqueeze(0),
            # TODO don't need
            torch.tensor(segment_ids, dtype=torch.long, device=self.device).unsqueeze(0),
            # This isn't used: to None?
            labels,
            torch.tensor(valid, dtype=torch.long, device=self.device).unsqueeze(0),
            torch.ones_like(labels),
        )
        return self.dataset

    def __iter__(self):
        yield self.dataset

    def __len__(self):
        return 1 if self.dataset else 0
