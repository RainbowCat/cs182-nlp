# import copy
import json

# import os
# import random
# import sys
# import time
# from pathlib import Path
from typing import Optional

# import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import tqdm
# from segtok import tokenizer
# from torch._C import LongTensor
# from torch.optim import lr_scheduler
# from torch.utils import data
from torch.utils.data import DataLoader, Dataset, random_split

# from torchvision import datasets, models, transforms
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import utils

nltk.download("punkt")


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
def get_tokenizer(args):
    if args.use_bert:
        model_name = "bert-base-cased"
    else:
        model_name = "xlnet-base-cased"
    return torch.hub.load("huggingface/pytorch-transformers", "tokenizer", model_name)


def tokenize_review(args, review):
    tokenizer = get_tokenizer(args)
    encodings = tokenizer(
        review['text'],
        add_special_tokens=True,
        max_length=args.max_len,
        return_token_type_ids=False,
        return_attention_mask=False,
        truncation=True,
        pad_to_max_length=True,
    )
    return encodings["input_ids"]


# padding
def pad_sequence(numerized, pad_index, to_length, beginning=False):
    pad = numerized[:to_length]
    if beginning:
        padded = [pad_index] * (to_length - len(pad)) + pad
    else:
        padded = pad + [pad_index] * (to_length - len(pad))
    mask = [w != pad_index for w in padded]
    return padded, mask


# formatting
def format_reviews(args, tokenizer, datatable: pd.DataFrame):
    encoded_reviews = []
    encoded_reviews_mask = []
    review_sentiments = []
    reviews_to_process = datatable[["review_id", "text", "stars"]]

    analyzer = SentimentIntensityAnalyzer()
    for i, review in reviews_to_process.iterrows():
        review_text = review["text"]
        numerized = tokenize_review(args, tokenizer, review_text)
        padded, mask = pad_sequence(numerized, 0, args.max_len)
        encoded_reviews.append(padded)
        encoded_reviews_mask.append(mask)

        # VADER
        sentiments = [
            analyzer.polarity_scores(s)["compound"]
            for s in nltk.tokenize.sent_tokenize(review_text)
        ]
        padded, _ = pad_sequence(sentiments, 0, args.max_len_vader)
        review_sentiments.append(padded)

    return (
        torch.LongTensor(encoded_reviews),  # text
        torch.FloatTensor(review_sentiments),  # sentiments
        torch.LongTensor(reviews_to_process["stars"].values),  # target
        torch.FloatTensor(encoded_reviews_mask),  # mask
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


# batching


class YelpDataset(Dataset):
    def __init__(self, args, data_path):
        print("creating dataset")
        self.data_path = data_path
        yelp_reviews_df = load_json(self.data_path, filter_function=lambda x: True)
        self.len = len(yelp_reviews_df)
        self.yelp_reviews = format_reviews(args, yelp_reviews_df)

    def __len__(self):
        return len(self.len)

    def __getitem__(self, idx):
        text, sentiments, target, mask = self.yelp_reviews
        return (text[idx], sentiments[idx], target[idx], mask[idx])


class YelpDataModule(pl.LightningDataModule):
    def __init__(self, args, dataset: Dataset):
        super().__init__()
        self.dataset = dataset
        self.batch_size = args.batch_size

    def setup(self, tokenizer, stage: Optional[str] = None):
        train_set, val_set, test_set = train_validate_test_split(self.dataset)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
