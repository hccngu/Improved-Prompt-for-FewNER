import logging
import os
import pickle
from multiprocessing import Pool
from typing import Tuple

import pandas as pd
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def preprocess_data(data):
    input_text, target_text, encoder_tokenizer, decoder_tokenizer, args = data

    input_text = encoder_tokenizer.encode(
        input_text, max_length=args.max_seq_length, pad_to_max_length=True, return_tensors="pt",
    )

    target_text = decoder_tokenizer.encode(
        target_text, max_length=args.max_seq_length, pad_to_max_length=True, return_tensors="pt"
    )
    return (torch.flatten(input_text), torch.flatten(target_text))


class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_tokenizer, decoder_tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir, args.model_name + "_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

        data = [
            (input_text, target_text, encoder_tokenizer, decoder_tokenizer, args)
            for input_text, target_text in zip(data["input_text"], data["target_text"])
        ]

        if args.use_multiprocessing:
            with Pool(args.process_count) as p:
                self.examples = list(
                    tqdm(
                        p.imap(preprocess_data, data, chunksize=args.multiprocessing_chunksize),
                        total=len(data),
                        disable=args.silent,
                    )
                )
        else:
            self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]

        logger.info(" Saving features into cached file %s", cached_features_file)
        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def preprocess_data_bart(data):
    input_text, target_text, binary_target_text, tokenizer, args = data

    input_ids = tokenizer.batch_encode_plus(
        [input_text], max_length=args.max_seq_length, padding='max_length', truncation=True, return_tensors="pt",
    )

    target_ids = tokenizer.batch_encode_plus(
        [target_text], max_length=args.max_seq_length, padding='max_length', truncation=True, return_tensors="pt"
    )

    if args.use_be is True:
        if args.te == 'te1':
            if target_text.split()[-4] == 'not':
                for i, index in enumerate(target_ids["input_ids"][0]):
                    if index not in [0, 1, 2]:
                        target_ids["input_ids"][0][i] = 1
        elif args.te in ['te2', 'te3', 'te4']:
            if target_text.split()[-2] == 'none':
                for i, index in enumerate(target_ids["input_ids"][0]):
                    if index not in [0, 1, 2]:
                        target_ids["input_ids"][0][i] = 1
        else:
            raise NotImplementedError

    if args.model_type == 'bart2decoder':
        if args.use_be:
            binary_target_ids = tokenizer.batch_encode_plus(
            [binary_target_text], max_length=args.max_seq_length, padding='max_length', truncation=True, return_tensors="pt"
            )
        else:
            output_words = target_text.split()
            is_index = len(output_words) - output_words[::-1].index('is') - 1
            entity_words = ' '.join(output_words[:is_index])
            if args.te in ['te2', 'te3', 'te4']:
                if output_words[-2] == 'none':
                    binary_target_text = entity_words + ' is not a named entity'
                else:
                    binary_target_text = entity_words + ' is a named entity'
            elif args.te == 'te1':
                if output_words[-4] == 'not':
                    binary_target_text = entity_words + ' is not a named entity'
                else:
                    binary_target_text = entity_words + ' is a named entity'
            else:
                raise NotImplementedError
            binary_target_ids = tokenizer.batch_encode_plus(
            [binary_target_text], max_length=args.max_seq_length, padding='max_length', truncation=True, return_tensors="pt"
            )
        return {
            "source_ids": input_ids["input_ids"].squeeze(),
            "source_mask": input_ids["attention_mask"].squeeze(),
            "target_ids": target_ids["input_ids"].squeeze(),
            "binary_target_ids": binary_target_ids["input_ids"].squeeze(),
        }
    else:
        return {
            "source_ids": input_ids["input_ids"].squeeze(),
            "source_mask": input_ids["attention_mask"].squeeze(),
            "target_ids": target_ids["input_ids"].squeeze(),
        }


class SimpleSummarizationDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(
            args.cache_dir, args.model_name + "_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)
        data = [
            (input_text, target_text, binary_target_text, tokenizer, args)
            for input_text, target_text, binary_target_text in zip(data["input_text"], data["target_text"], data["O-entity"])
        ]

        if args.use_multiprocessing:
            with Pool(args.process_count) as p:
                self.examples = list(
                    tqdm(
                        p.imap(preprocess_data_bart, data, chunksize=args.multiprocessing_chunksize),
                        total=len(data),
                        disable=args.silent,
                    )
                )
        else:
            self.examples = [preprocess_data_bart(d) for d in tqdm(data, disable=args.silent)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
