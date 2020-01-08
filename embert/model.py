#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Defines the :class:`TokenClassifier` class."""

import torch
from transformers import BertForTokenClassification


# TODO do we REALLY need to subclass BertForTokenClassification?!
class TokenClassifier(BertForTokenClassification):
    # TODO check whether the arguments in the second row are the same as in
    # BertForTokenClassification, but with different names (see below)
    # def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
    #             position_ids=None, head_mask=None, inputs_embeds=None, labels=None)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels=None, valid_ids=None, attention_mask_label=None):
        sequence_output = self.bert(input_ids, attention_mask, token_type_ids,
                                    head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        # TODO: not necessarily CUDA...
        valid_output = torch.zeros(batch_size, max_len, feat_dim,
                                   dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # TODO: why is this zeroed?
            attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            return (loss, logits)
        else:
            # TODO Output should also be different: a tuple, whose content
            # depends on how the function was called
            return (logits,)
