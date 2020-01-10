#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The actual emtsv interface."""

class EmBERT:
    def __init__(self, task='ner', source_fields=None, target_fields=None):
        # Field names for e-magyar TSV
        if source_fields is None:
            source_fields = set()

        if target_fields is None:
            target_fields = []

        self.source_fields = source_fields
        self.target_fields = target_fields

    def process_sentence(self, sen, field_names):
        for tok in sen:
            tok.append('O')
        return sen

    @staticmethod
    def prepare_fields(field_names):
        return [field_names['form']]
