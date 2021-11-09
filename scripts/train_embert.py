#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trains a token classification model. Original code taken from
https://github.com/kamalkraj/BERT-NER.

The script can both train and evaluate a token classifier. In production, a
"reverse" Viterbi model is also applied to the result to prevent the emission
of illegal tag sequences; however, the evaluation in this script does not
make use of this state transition model.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import json
import logging
import os
import random
import sys

import numpy as np
from seqeval.metrics import classification_report
import torch
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler
# from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (AdamW, BertConfig,
                          BertTokenizer, get_linear_schedule_with_warmup)

from embert.data_format import all_formats, get_format_reader
from embert.extract_transitions import default_transitions
from embert.evaluate import Evaluator
from embert.model import TokenClassifier
from embert.data_wrapper import DataWrapper, DatasetWrapper
from embert.processors import DataProcessor, DataSplit
from embert.viterbi import ReverseViterbi


# tqdm that prints the progress bar to stdout. This helps keeping the log
# clean
otqdm = partial(tqdm, file=sys.stdout)
otrange = partial(trange, file=sys.stdout)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True,
                        help='The input data dir. Should contain the .tsv '
                             'files (or other data files) for the task, and '
                             'the label should be in the last column.')
    parser.add_argument("--bert_model",
                        help='The pre-trained BERT model to base the '
                             'classifier on. Can be a directory with a '
                             'PyTorch checkpoint or any named Huggingface '
                             'model. The recommended model is '
                             'SZTAKI-HLT/hubert-base-cc. Required for '
                             'training, but not for evaluation.')
    parser.add_argument('--task_name', required=True,
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
    parser.add_argument("--use_viterbi", action='store_true',
                        help='Whether to use the (reverse) Viterbi algorithm '
                             'with default transition probabilities on top of '
                             'the token classifier for evaluation (validation '
                             'and test).')
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
                             'learning rate warmup for. E.g., 0.1 = 10%% of '
                             'training.')
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help='Weight decay if we apply some.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help='Epsilon for Adam optimizer.')
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help='Maximum gradient norm.')
    parser.add_argument("--no_cuda", action='store_true',
                        help='Whether not to use CUDA when available')
    parser.add_argument("--gpu", type=int, default=0,
                        help='the index of the GPU to use (0).')
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
    if args.do_train and not args.bert_model:
        parser.error('--bert_model is required for training.')
    if args.gradient_accumulation_steps < 1:
        parser.error('--gradient_accumulation_steps should be >= 1')

    return args


def real_loss(loss, n_gpu):
    """Averages the loss when multiple GPUs are used."""
    return loss.mean() if n_gpu > 1 else loss


class Trainer:
    """Runs the training."""
    def __init__(self, model: nn.Module, train_wrapper: DataWrapper,
                 valid_wrapper: DataWrapper, device: torch.device,
                 epochs: float = 3, batch_size: int = 32,
                 learning_rate: float = 5e-5, warmup_proportion: float = 0.1,
                 adam_epsilon: float = 1e-8, weight_decay: float = 0.01,
                 max_grad_norm: float = 1.0, gradient_accumulation_steps: int = 1,
                 fp16: bool = False, fp16_opt_level: str = 'O1',
                 viterbi: ReverseViterbi = None):
        """
        Initializes the objects required for training (optimizer, scheduler,
        etc.)

        :param model: the model to train.
        :param train_wrapper: the :class:`DataWrapper` for the train data.
        :param valid_wrapper: the :class:`DataWrapper` for the validation data.
        :param device: the :mod:`torch` device the training should run on.
        :param epochs: the total number of training epochs to perform.
        :param batch_size: batch size for training.
        :param learning_rate: the initial learning rate for Adam.
        :param warmup_proportion: proportion of training to perform linear
                                  learning rate warmup for.
        :param adam_epsilon: Epsilon for Adam optimizer.
        :param weight_decay: weight decay if we apply some.
        :param max_grad_norm: maximum gradient norm.
        :param gradient_accumulation_steps: number of updates steps to
                                            accumulate before performing a
                                            backward/update pass.
        :param fp16: whether to use 16-bit float precision instead of 32-bit.
        :param fp16_opt_level: for fp16: Apex AMP optimization level selected.
        :param viterbi: a (reverse) Viterbi model for validation.
        """
        self.model = model
        self.train_wrapper = train_wrapper
        self.valid_wrapper = valid_wrapper
        self.device = device
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.viterbi = viterbi

        self.global_step = 0

        # Set up model on device, optimizer, etc.
        model.to(device)

        # TODO: same as len(train_wrapper)?
        num_batches = train_wrapper.num_examples / batch_size
        opt_steps_per_epoch = num_batches / gradient_accumulation_steps
        self.num_train_optimization_steps = self.epochs * opt_steps_per_epoch
        # if local_rank != -1:
        #     self.num_train_optimization_steps //= torch.distributed.get_world_size()

        # Optimization
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer
                        if not any(nd in name for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [param for name, param in param_optimizer
                        if any(nd in name for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        warmup_steps = int(warmup_proportion * self.num_train_optimization_steps)
        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=learning_rate, eps=adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=self.num_train_optimization_steps
        )

        # 16-bit floating-point precision
        if fp16:
            try:
                from apex import amp
                self.amp = amp
            except ImportError:
                raise ImportError(
                    'Please install apex from https://www.github.com/nvidia/apex '
                    'to use fp16 training.'
                )
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=fp16_opt_level)

        # # multi-gpu training (should be after apex fp16 initialization)
        # if n_gpu > 1:
        #     self.model = torch.nn.DataParallel(self.model)

        # if local_rank != -1:
        #     self.model = torch.nn.parallel.DistributedDataParallel(
        #         self.model, device_ids=[local_rank],
        #         output_device=local_rank, find_unused_parameters=True
        #     )

    def train(self):
        logging.info('***** Running training *****')
        logging.info(f'  Num examples = {self.train_wrapper.num_examples}')
        logging.info(f'  Batch size = {self.train_wrapper.batch_size}')
        logging.info(f'  Num steps = {self.num_train_optimization_steps}')

        stats = {'train_loss': 0, 'num_examples': 0, 'num_steps': 0}
        for epoch in otrange(int(self.epochs), desc='Epoch'):
            self.train_step(stats)

            logging.info(f'Validation for epoch {epoch}:')
            with save_random_state():
                loss, report = evaluate(self.model, self.valid_wrapper,
                                        self.viterbi)
                # loss = real_loss(loss, self.n_gpu)
            logging.debug(f'Validation loss in epoch {epoch}: {loss}')
            for i, param_group in enumerate(self.optimizer.param_groups):
                logging.info(f'LR {i}: {param_group["lr"]}')
            # new_lr = sum(pg['lr'] for pg in self.optimizer.param_groups) / \
            #     len(self.optimizer.param_groups)
            # if new_lr < last_lr * 0.9:
            #     ...

    def train_step(self, stats):
        """Runs a single epoch of training."""
        self.model.train()
        for step, (
            input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask
        ) in enumerate(otqdm(self.train_wrapper, desc='Iteration')):
            loss, _ = self.model(input_ids, segment_ids, input_mask,
                                 label_ids, valid_ids, l_mask)
            loss = real_loss(loss, 1)  # n_gpu
            if self.gradient_accumulation_steps > 1:
                loss /= self.gradient_accumulation_steps

            if self.fp16:
                with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.amp.master_params(self.optimizer), self.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_grad_norm)

            stats['train_loss'] += loss.item()
            stats['num_examples'] += input_ids.size(0)
            stats['num_steps'] += 1
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                self.global_step += 1

    def get_real_model(self):
        """Returns the model without all the (e.g. distributed) wrappers."""
        return self.model.module if hasattr(self.model, 'module') else self.model


def evaluate(model: nn.Module, wrapper: DataWrapper,
             viterbi: ReverseViterbi = None):
    """Runs a full evaluation loop."""
    logging.info(f'***** Running evaluation: {wrapper.split.value} *****')
    logging.info(f'  Num examples = {wrapper.num_examples}')
    logging.info(f'  Batch size = {wrapper.batch_size}')
    logging.info(f'  Num steps = {wrapper.num_steps}')

    eval = Evaluator(model, wrapper, viterbi,
                     partial(otqdm, desc=f'Evaluating {wrapper.split.value}...'))
    results = eval.evaluate()

    if results.num_zero_labels > 0:
        logging.warning(f'Predicted {results.num_zero_labels} zero labels!')

    report = classification_report(results.y_true, results.y_pred, digits=4)
    logging.info('***** Eval results *****')
    logging.info(f'\n{report}')
    return results.get_loss(), report


@contextmanager
def save_random_state():
    """
    Saves the states of all random number generators prior to executing a code
    block and loads them back afterwards. Useful to add evaluation steps
    without compromising the reproducability of training.
    """
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

    logging.info(f'Args: {args}')

    format_reader = get_format_reader(args.data_format)
    processor = DataProcessor(args.data_dir, format_reader)

    if args.use_viterbi:
        viterbi = ReverseViterbi(*default_transitions(processor.get_labels()))
    else:
        viterbi = None

    # TODO is this right?
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        # n_gpu = torch.cuda.device_count()
    else:
        # TODO why both?
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda", args.gpu)
        # # Initializes the distributed backend which will take care of
        # # sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
    logging.info(f'Device information: device: {device}, '
                 f'gpu: {args.gpu}, 16-bits training: {args.fp16}')

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if args.do_train:
            raise ValueError(f'Output directory ({args.output_dir}) '
                             f'already exists and is not empty.')
        else:
            model_dir = args.output_dir
    else:
        model_dir = args.bert_model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Make sure the results are reproducible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # See https://pytorch.org/docs/stable/notes/randomness.html
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_labels = len(processor.get_labels()) + 1

    tokenizer = BertTokenizer.from_pretrained(
        model_dir, do_lower_case=args.do_lower_case,
        do_basic_tokenize=False)  # In quntoken we trust

    if args.do_train:
        # train_examples = processor.get_train_examples()
        train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        # # TODO: understand train_batch_size and clean this up
        # num_train_optimization_steps = args.num_train_epochs * int(
        #     len(train_examples) / train_batch_size /
        #     args.gradient_accumulation_steps
        # )
        # logging.info(f'Using a train batch size of {train_batch_size}.')

        # if args.local_rank != -1:
        #     num_train_optimization_steps //= torch.distributed.get_world_size()

    try:
        if args.do_train:
            # # TODO understand distributed execution
            # # Make sure only the first process in distributed training will
            # # download model & vocab
            # if args.local_rank not in [-1, 0]:
            #     torch.distributed.barrier()

            # Prepare model
            config = BertConfig.from_pretrained(
                model_dir,
                # os.path.join(args.bert_model, 'bert_config.json'),
                num_labels=num_labels, finetuning_task=args.task_name
            )
            model = TokenClassifier.from_pretrained(
                model_dir, from_tf=False, config=config)

            # # Make sure only the first process in distributed training will
            # # download model & vocab
            # if args.local_rank == 0:
            #     torch.distributed.barrier()

            # Create the data wrappers
            train_wrapper = DatasetWrapper(
                processor, DataSplit.TRAIN,
                RandomSampler,
                train_batch_size, args.max_seq_length, tokenizer, device
            )
            valid_wrapper = DatasetWrapper(
                processor, DataSplit.VALID, SequentialSampler, args.eval_batch_size,
                args.max_seq_length, tokenizer, device
            )

            trainer = Trainer(
                model, train_wrapper, valid_wrapper, device,
                args.num_train_epochs, train_batch_size,
                args.learning_rate, args.warmup_proportion,
                args.adam_epsilon, args.weight_decay,
                args.max_grad_norm, args.gradient_accumulation_steps,
                args.fp16, args.fp16_opt_level, viterbi
            )
            trainer.train()

            # Save the configuration and vocabulary associated with the
            # trained model.
            tokenizer.save_pretrained(args.output_dir)
            trainer.get_real_model().save_pretrained(args.output_dir)
            model_config = {
                'bert_model': args.bert_model, 'do_lower': args.do_lower_case,
                'max_seq_length': args.max_seq_length,
                'num_labels': num_labels,
                'labels': train_wrapper.get_labels()
            }
            with open(os.path.join(args.output_dir, 'model_config.json'), 'w') as f:
                json.dump(model_config, f)

            del train_wrapper, valid_wrapper, trainer
        else:
            # No do_train:
            # Load a trained model and vocabulary that you have fine-tuned
            model = TokenClassifier.from_pretrained(model_dir)
            tokenizer = BertTokenizer.from_pretrained(
                model_dir, do_lower_case=args.do_lower_case,
                do_basic_tokenize=False)  # In quntoken we trust
    except KeyboardInterrupt:
        logging.info('Stopping training...')

    model.to(device)

    if args.do_eval:
        test_wrapper = DatasetWrapper(
            processor, DataSplit.TEST, SequentialSampler, args.eval_batch_size,
            args.max_seq_length, tokenizer, device
        )
        with save_random_state():
            _, report = evaluate(model, test_wrapper, viterbi)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, 'wt') as writer:
            writer.write(report)


if __name__ == "__main__":
    main()
