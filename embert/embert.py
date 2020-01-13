#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The actual emtsv interface."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import (BertTokenizer)
import yaml

from .data_wrapper import SentenceWrapper
from .evaluate import predict
from .model import TokenClassifier

class EmBERT:
    def __init__(self, task='ner', source_fields=None, target_fields=None):
        # Field names for e-magyar TSV
        if source_fields is None:
            source_fields = set()

        if target_fields is None:
            target_fields = []

        self.source_fields = source_fields
        self.target_fields = target_fields

        self.config = {'no_cuda': False, 'max_seq_length': 512}
        self.config.update(self.read_config(task))
        self.load_model()

    def read_config(self, config) -> Dict[str, Any]:
        """Reads the YAML configuration file from the ``configs`` directory."""
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        with open(config_dir / f'{config}.yaml') as inf:
            return yaml.load(inf, Loader=yaml.SafeLoader)

    def load_model(self):
        """
        Loads the model specified by the configuration from the configuration
        folder. If it doesn't exist, tries to download it.
        """
        models_dir = Path(__file__).resolve().parents[1] / 'models'
        try:
            model_dir = models_dir / self.config['model']
        except KeyError:
            raise ValueError('Key "model" is missing from the configuration.')

        if not os.path.isdir(model_dir):
            # TODO: download
            pass

        try:
            cuda = torch.cuda.is_available() and not self.config['no_cuda']
            device = torch.device('cuda' if cuda else 'cpu')

            tokenizer = BertTokenizer.from_pretrained(
                model_dir, do_lower_case=False,
                do_basic_tokenize=False)  # In quntoken we trust
            self.model = TokenClassifier.from_pretrained(model_dir)
            self.model.to(device)
            with open(model_dir / 'model_config.json', 'rb') as inf:
                model_config = json.load(inf)
            self.wrapper = SentenceWrapper(model_config['labels'],
                                           self.config['max_seq_length'],
                                           tokenizer, device)
        except Exception as e:
            raise ValueError(f'Could not load model {self.config["model"]}: {e}')

    def process_sentence(self, sen, field_names):
        self.wrapper.set_sentence([tok[field_names[0]] for tok in sen])
        classes = predict(self.model, self.wrapper)[0]
        for tok, cls in zip(sen, classes):
            tok.append(cls)
        return sen

    @staticmethod
    def prepare_fields(field_names):
        return [field_names['form']]
