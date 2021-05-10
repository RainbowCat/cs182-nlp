import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from segtok import tokenizer
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import *

nltk.download("punkt")


def batch_to_torch_long(*batches):
    if len(batches) == 1:
        return torch.LongTensor(batches[0])
    return [torch.LongTensor(batch) for batch in batches]


def batch_to_torch_float(*batches):
    if len(batches) == 1:
        return torch.FloatTensor(batches[0])
    return [torch.FloatTensor(batch) for batch in batches]


def load_json(file_path, filter_function=lambda x: True):
    """
    file_path - full path of the file to read from
    filter_function - a data selection function, returns True to ADD a data point
    """
    result = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                json_line = json.loads(line)
                if not filter_function(json_line):
                    # Disallow via opposite of allow
                    continue
                result.append(
                    json_line
                )  # one data point represented as a dictionary per line
        return pd.DataFrame.from_records(result)
        # return result
    except IOError:
        print(f"cannot open {file_path}")
        return None


# tokenizing
def tokenize(data):
    """
    data - an iterable of sentences
    """
    token_set = set()
    i = 0
    for sentences in data:
        if i % 1000 == 0:
            print(i, end=", " if i % 15000 != 0 else "\n")
        tokenized = tokenizer.word_tokenizer(sentences.lower())
        for token in tokenized:
            token_set.add(token)
        i += 1
    return token_set


def tokenize_review(args, tokenizer, review_text):
    encodings = tokenizer.encode_plus(
        review_text,
        add_special_tokens=True,
        max_length=args.max_len,
        return_token_type_ids=False,
        return_attention_mask=False,
        truncation=True,
        pad_to_max_length=False,
    )
    return encodings.get("input_ids", [])


# padding
def pad_sequence(numerized, pad_index, to_length, beginning=True):
    pad = numerized[:to_length]
    if beginning:
        padded = [pad_index] * (to_length - len(pad)) + pad
    else:
        padded = pad + [pad_index] * (to_length - len(pad))
    mask = [w != pad_index for w in padded]
    return padded, mask


# batching
batch_to_torch = lambda b_in, b_targets, b_mask: (
    torch.LongTensor(b_in),
    torch.LongTensor(b_in),
    torch.FloatTensor(b_mask),
)

# formatting


def format_reviews(args, tokenizer, datatable, indices=None, task_bar=False):
    encoded_reviews = []
    encoded_reviews_mask = []
    review_sentiment = []
    reviews_to_process = datatable[["review_id", "text", "stars"]]
    # display(reviews_to_process)
    if indices is not None:
        reviews_to_process = reviews_to_process.iloc[indices]

    review_iterator = reviews_to_process.iterrows()
    if task_bar:
        review_iterator = tqdm.tqdm(
            reviews_to_process.iterrows(), total=reviews_to_process.shape[0]
        )

    analyzer = SentimentIntensityAnalyzer()
    for i, review in review_iterator:
        # Tokenize by TOKENIZER
        review_text = review["text"]
        numerized = tokenize_review(args, tokenizer, review_text)
        padded, mask = pad_sequence(numerized, 0, args.max_len)
        encoded_reviews.append(padded)
        encoded_reviews_mask.append(mask)

        # VADER
        sentence_list = nltk.tokenize.sent_tokenize(review_text)
        review_sentiment_sentence = []

        for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            review_sentiment_sentence.append(vs["compound"])
        padded, _ = pad_sequence(review_sentiment_sentence, 0, args.max_len_vader)
        review_sentiment.append(padded)

    torch_encoded_reviews, torch_encoded_reviews_target = batch_to_torch_long(
        encoded_reviews, reviews_to_process["stars"].values
    )
    torch_encoded_reviews_mask, torch_review_sentiment = batch_to_torch_float(
        encoded_reviews_mask, review_sentiment
    )
    return (
        torch_encoded_reviews,
        torch_encoded_reviews_target,
        torch_encoded_reviews_mask,
        torch_review_sentiment,
    )


# split up dataset
# https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
def train_validate_test_split(df, train_percent=0.6, validate_percent=0.2, seed=0):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    # m = df.size
    m = len(df.index)

    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end

    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]

    return train, validate, test


class YelpDataset(Dataset):
    def __init__(self, data_path):
        self.yelp_reviews = load_json(self.data_dir, filter_function=lambda x: True)

    def __len__(self):
        return len(self.yelp_reviews)

    def __getitem__(self, idx):
        return self.yelp_reviews.iloc["text"][idx]


class YelpDataModule(pl.LightningDataModule):
    def __init__(self, args, data_path: str):
        super().__init__()
        self.data_path = data_path
        self.batch_size = args.batch_size

    def setup(self, tokenizer, stage: Optional[str] = None):
        full_dataset = YelpDataset(self.data_path)
        self.train_set, self.val_set, self.test_set = train_validate_test_split(
            full_dataset
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
