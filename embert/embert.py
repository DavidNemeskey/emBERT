#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The actual emtsv interface."""

import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import torch
from transformers import BertTokenizer
import yaml

from .data_wrapper import SentenceWrapper
from .extract_transitions import default_transitions
# from .evaluate import predict
from .model import TokenClassifier
from .viterbi import ReverseViterbi

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
        self._load_model()

    def read_config(self, config) -> Dict[str, Any]:
        """Reads the YAML configuration file from the ``configs`` directory."""
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        with open(config_dir / f'{config}.yaml') as inf:
            return yaml.load(inf, Loader=yaml.SafeLoader)

    def _load_model(self):
        """
        Loads the model specified by the configuration from the configuration
        folder.
        """
        # "Logging" a'la HunTag3
        print(f'Loading BERT {self.config["model"]} model...',
              end='', file=sys.stderr, flush=True)
        models_dir = Path(__file__).resolve().parents[1] / 'models'
        try:
            model_dir = models_dir / self.config['model']
            if not model_dir.is_dir():
                raise ValueError(
                    f'Model {self.config["model"]} not found in {models_dir}. '
                    f'Please make sure you download it via emtsv\'s '
                    f'download_models.py.')
        except KeyError:
            raise ValueError('Key "model" is missing from the configuration.')

        try:
            tokenizer, self.model = self._load_model_from_disk(str(model_dir))

            cuda = torch.cuda.is_available() and not self.config['no_cuda']
            device = torch.device('cuda' if cuda else 'cpu')

            self.model.to(device)
            with open(model_dir / 'model_config.json', 'rb') as inf:
                model_config = json.load(inf)
            self.wrapper = SentenceWrapper(model_config['labels'],
                                           self.config['max_seq_length'],
                                           tokenizer, device)
            print('done', file=sys.stderr, flush=True)
        except Exception as e:
            raise ValueError(f'Could not load model {self.config["model"]}: {e}')

        init_stats, transitions = default_transitions(self.wrapper.get_labels())
        self.viterbi = ReverseViterbi(init_stats, transitions)

    def _load_model_from_disk(self, model_dir: str) -> Tuple[
        BertTokenizer, TokenClassifier
    ]:
        """Loads the tokenizer and the classifier from _model_dir_."""
        tokenizer = BertTokenizer.from_pretrained(
            model_dir, do_lower_case=False,
            do_basic_tokenize=False)  # In quntoken we trust
        model = TokenClassifier.from_pretrained(model_dir)
        return tokenizer, model

    def process_sentence(self, sen, field_names):
        self.wrapper.set_sentence([tok[field_names[0]] for tok in sen])
        classes = predict(self.model, self.wrapper, self.viterbi)[0]
        for tok, cls in zip(sen, classes):
            tok.append(cls)
        return sen

    @staticmethod
    def prepare_fields(field_names):
        return [field_names['form']]
