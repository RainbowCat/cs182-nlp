import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from keras.preprocessing.sequence import pad_sequences
from segtok import tokenizer
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import *


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
                result.append(json_line)  # each line is one data point dictionary
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


def tokenize_review(tokenizer, review_text):
    encodings = tokenizer.encode_plus(
        review_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
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
def format_reviews(tokenizer, datatable, indices=None):
    encoded_reviews = []
    encoded_reviews_mask = []
    reviews_to_process = datatable[["text", "stars"]]
    if indices is not None:
        reviews_to_process = reviews_to_process.iloc[indices]

    for review_text in reviews_to_process["text"]:
        numerized = tokenize_review(tokenizer, review_text)
        padded, mask = pad_sequence(numerized, 0, MAX_LEN)
        encoded_reviews.append(padded)
        encoded_reviews_mask.append(mask)

    (
        torch_encoded_reviews,
        torch_encoded_reviews_target,
        torch_encoded_reviews_mask,
    ) = batch_to_torch(
        encoded_reviews, reviews_to_process["stars"], encoded_reviews_mask
    )
    return (
        torch_encoded_reviews,
        torch_encoded_reviews_target,
        torch_encoded_reviews_mask,
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