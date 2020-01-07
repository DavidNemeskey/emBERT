#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Trains a NER model."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import os
from functools import partial
import random
import sys
from typing import Dict, List, Type

import numpy as np
from seqeval.metrics import classification_report
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, Sampler,
                              SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (AdamW, BertConfig,
                          BertForTokenClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)

from data_format import all_formats, get_format_reader
from processors import all_processors, get_processor, DataProcessor, DataSplit


# tqdm that prints the progress bar to stdout. This helps keeping the log
# clean
otqdm = partial(tqdm, file=sys.stdout)
otrange = partial(trange, file=sys.stdout)


class Ner(BertForTokenClassification):
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
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
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


@dataclass
class InputFeatures:
    """A single set of features of data."""
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


def convert_examples_to_features(examples, label_list,
                                 max_seq_length, tokenizer):
    """Loads a data file into a list of :class:`InputFeatures`s."""
    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for ex_index, example in enumerate(examples):
        tokens = []
        labels = example.labels[:]
        valid = []
        for i, word in enumerate(example.text_a):
            word_tokens = tokenizer.tokenize(word)
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
        label_ids = [label_map[l] for l in ['[CLS]'] + labels + ['[SEP]']]
        label_mask = [1] * len(label_ids)
        valid.insert(0, 1)
        valid.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        if ex_index < 5:
            logging.info('*** Example ***')
            logging.info(f'guid: {example.guid}')
            logging.info(f'tokens: {" ".join(tokens)}')
            logging.info(f'input_ids: {" ".join(str(x) for x in input_ids)}')
            logging.info(f'input_mask: {" ".join(str(x) for x in input_mask)}')
            logging.info(f'segment_ids: {" ".join(str(x) for x in segment_ids)}')
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


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True,
                        help='The input data dir. Should contain the .tsv '
                             'files (or other data files) for the task.')
    parser.add_argument("--bert_model", required=True,
                        help='Bert pre-trained model selected in the list: '
                             'bert-base-uncased, bert-large-uncased, '
                             'bert-base-cased, bert-large-cased, '
                             'bert-base-multilingual-uncased, '
                             'bert-base-multilingual-cased, '
                             'bert-base-chinese.')
    parser.add_argument('--task_name', required=True, choices=all_processors(),
                        help='The name of the task to train.')
    parser.add_argument('--data_format', required=True, choices=all_formats(),
                        help='The data format of the input files.')
    parser.add_argument("--output_dir", required=True,
                        help='The output directory where the model '
                             'predictions and checkpoints will be written.')

    # Other parameters
    parser.add_argument("--cache_dir", default='', type=str,
                        help='To store the pre-trained models downloaded from s3')
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help='The maximum total input sequence length after '
                             'WordPiece tokenization. Sequences longer than '
                             'this will be truncated, and sequences shorter '
                             'than this will be padded.')
    parser.add_argument("--do_train", action='store_true',
                        help='Whether to run training.')
    parser.add_argument("--do_eval", action='store_true',
                        help='Whether to run eval on the test set.')
    parser.add_argument("--do_lower_case", action='store_true',
                        help='Set this flag if you are using an uncased model.')
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help='Total batch size for training.')
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help='Total batch size for eval.')
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help='Total number of training epochs to perform.')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help='Proportion of training to perform linear '
                             'learning rate warmup for. E.g., 0.1 = 10% of '
                             'training.')
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help='Weight deay if we apply some.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help='Epsilon for Adam optimizer.')
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help='Max gradient norm.')
    parser.add_argument("--no_cuda", action='store_true',
                        help='Whether not to use CUDA when available')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help='local_rank for distributed training on gpus')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for initialization')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before '
                             'performing a backward/update pass.')
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use 16-bit float precision instead '
                             'of 32-bit')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        choices=['O0', 'O1', 'O2', 'O3'],
                        help='For fp16: Apex AMP optimization level selected. '
                             'See details at '
                             'https://nvidia.github.io/apex/amp.html')
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help='Loss scaling to improve fp16 numeric stability. '
                             'Only used when fp16 set to True. '
                             '0 (default value): dynamic loss scaling.'
                             'Positive power of 2: static loss scaling value.')
    args = parser.parse_args()

    if not args.do_train and not args.do_eval:
        parser.error('At least one of `do_train` or `do_eval` must be True.')
    if args.gradient_accumulation_steps < 1:
        parser.error('--gradient_accumulation_steps should be >= 1')

    return args


def real_loss(loss, n_gpu):
    """Averages the loss when multiple GPUs are used."""
    return loss.mean() if n_gpu > 1 else loss


class DataWrapper:
    def __init__(self, processor: DataProcessor, split: DataSplit,
                 sampler_cls: Type[Sampler], batch_size: int,
                 max_seq_length: int, tokenizer: BertTokenizer,
                 device: torch.device):
        self.processor = processor
        self.split = split
        self.device = device

        examples = processor.get_examples(split)
        features = convert_examples_to_features(
            examples, processor.get_labels(), max_seq_length, tokenizer)

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

    def __iter__(self):
        for batch in self.dataloader:
            # Crop tensors to actual length
            actual_seq_length = batch[1].sum(dim=1).max()
            yield (t[:, :actual_seq_length].contiguous() for t in batch)
        # yield from self.dataloader

    def __len__(self):
        # Might not work for iterable datasets
        return len(self.dataloader)


def train(model: nn.Module, processor: DataProcessor,
          split: DataSplit, max_seq_length: int, tokenizer: BertTokenizer,
          batch_size: int, label_map: Dict[str, int], device):
    """Runs a single epoch of training."""
    raise NotImplementedError('train() is not implemented yet')


def evaluate(model: nn.Module, wrapper: DataWrapper, label_map: Dict[str, int]):
    """Runs a full evaluation loop."""
    logging.info(f'***** Running evaluation: {wrapper.split.value} *****')
    logging.info(f'  Num examples = {wrapper.num_examples}')
    logging.info(f'  Batch size = {wrapper.batch_size}')
    logging.info(f'  Num steps = {wrapper.num_steps}')

    model.eval()
    y_true = []
    y_pred = []
    for (
        input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask
    ) in otqdm(wrapper, desc=f'Evaluating {wrapper.split.value}...'):
        with torch.no_grad():
            loss, logits = model(input_ids, segment_ids, input_mask, labels=label_ids,
                                 valid_ids=valid_ids, attention_mask_label=l_mask)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        num_zero_labels = 0
        label_list = list(label_map.values())
        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    # It can happen, that logits[i][j] is 0, which is not
                    # in label_map. In that case, we return a random label
                    if logits[i][j] != 0:
                        temp_2.append(label_map[logits[i][j]])
                    else:
                        num_zero_labels += 1
                        temp_2.append(random.choice(label_list))

    if num_zero_labels > 0:
        logging.warning(f'Predicted {num_zero_labels} zero labels!')

    report = classification_report(y_true, y_pred, digits=4)
    logging.info('***** Eval results *****')
    logging.info(f'\n{report}')
    return loss, report


@contextmanager
def save_random_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


def main():
    args = parse_arguments()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG
    )

    # TODO is this right?
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.info(f'Device informantion: device: {device} n_gpu: {n_gpu}, '
                 f'distributed training: {args.local_rank != -1}, '
                 f'16-bits training: {args.fp16}')

    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    logging.info(f'Using a train batch size of {train_batch_size}.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # See https://pytorch.org/docs/stable/notes/randomness.html
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(f'Output directory ({args.output_dir}) '
                         f'already exists and is not empty.')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    format_reader = get_format_reader(args.data_format)
    processor = get_processor(args.task_name)(args.data_dir, format_reader)
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples()
        # TODO: understand train_batch_size and clean this up
        num_train_optimization_steps = args.num_train_epochs * int(
            len(train_examples) / train_batch_size /
            args.gradient_accumulation_steps
        )
        if args.local_rank != -1:
            num_train_optimization_steps //= torch.distributed.get_world_size()

    # TODO understand distributed execution
    # Make sure only the first process in distributed training will
    # download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Prepare model
    config = BertConfig.from_pretrained(
        args.bert_model,
        # os.path.join(args.bert_model, 'bert_config.json'),
        num_labels=num_labels, finetuning_task=args.task_name
    )
    model = Ner.from_pretrained(args.bert_model,
                                from_tf=False, config=config)

    # Make sure only the first process in distributed training will
    # download model & vocab
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(device)

    # Optimization
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [param for name, param in param_optimizer
                    if not any(nd in name for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [param for name, param in param_optimizer
                    if any(nd in name for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    last_lr = args.learning_rate
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=num_train_optimization_steps
    )

    # 16-bit floating-point precision
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                'Please install apex from https://www.github.com/nvidia/apex '
                'to use fp16 training.'
            )
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True
        )

    # The effective number of steps (i.e. minibatches / backpropagations)
    global_step = 0
    logging.debug(f'label list {label_list}')
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    logging.info(f'Label map: {label_map}')
    try:
        if args.do_train:
            if args.local_rank == -1:
                train_sampler = RandomSampler
            else:
                train_sampler = DistributedSampler

            train_wrapper = DataWrapper(
                processor, DataSplit.TRAIN, train_sampler, train_batch_size,
                args.max_seq_length, tokenizer, device
            )
            valid_wrapper = DataWrapper(
                processor, DataSplit.VALID, SequentialSampler, args.eval_batch_size,
                args.max_seq_length, tokenizer, device
            )

            logging.info(f'***** Running training *****')
            logging.info(f'  Num examples = {train_wrapper.num_examples}')
            logging.info(f'  Batch size = {train_batch_size}')
            logging.info(f'  Num steps = {num_train_optimization_steps}')

            model.train()
            for epoch in otrange(int(args.num_train_epochs), desc='Epoch'):
                logging.info(f'Epoch {epoch}.')
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, (
                    input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask
                ) in enumerate(otqdm(train_wrapper, desc='Iteration')):
                    loss, _ = model(input_ids, segment_ids, input_mask,
                                    label_ids, valid_ids, l_mask)
                    loss = real_loss(loss, n_gpu)
                    if args.gradient_accumulation_steps > 1:
                        loss /= args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                       args.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       args.max_grad_norm)

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1
                else:
                    logging.info(f'Validation for epoch {epoch}:')
                    with save_random_state():
                        loss, report = evaluate(model, valid_wrapper, label_map)
                        loss = real_loss(loss, n_gpu)
                    logging.debug(f'Validation loss in epoch {epoch}: {loss}')
                    for i, param_group in enumerate(optimizer.param_groups):
                        logging.info(f'LR {i}: {param_group["lr"]}')
                    new_lr = sum(pg['lr'] for pg in optimizer.param_groups) / len(optimizer.param_groups)
                    if new_lr < last_lr * 0.9:
                        logging.info(f'new_lr: {new_lr}')
                        if new_lr < 1e-12:
                            logging.info(f'Stopping: LR fell below 1e-10')
                            break

            # Save the configuration and vocabulary associated with the
            # trained model.
            tokenizer.save_pretrained(args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            model_config = {
                'bert_model': args.bert_model, 'do_lower': args.do_lower_case,
                'max_seq_length': args.max_seq_length,
                'num_labels': len(label_list) + 1,
                'label_map': label_map
            }
            with open(os.path.join(args.output_dir, 'model_config.json'), 'w') as f:
                json.dump(model_config, f)
        else:
            # No do_train:
            # Load a trained model and vocabulary that you have fine-tuned
            model = Ner.from_pretrained(args.output_dir)
            tokenizer = BertTokenizer.from_pretrained(
                args.output_dir, do_lower_case=args.do_lower_case)
    except KeyboardInterrupt:
        logging.info('Stopping training...')

    model.to(device)

    # TODO: do it regardless of local_rank
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        test_wrapper = DataWrapper(
            processor, DataSplit.TEST, SequentialSampler, args.eval_batch_size,
            args.max_seq_length, tokenizer, device
        )
        _, report = evaluate(model, test_wrapper, label_map)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, 'wt') as writer:
            writer.write(report)


if __name__ == "__main__":
    main()
