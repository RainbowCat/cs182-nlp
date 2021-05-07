#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from itertools import chain, product
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence

import os
import sys
import argparse
from pathlib import Path

import json
import pandas as pd
import random

import torch
from segtok import tokenizer
from keras.preprocessing.sequence import pad_sequences
import tqdm

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from transfomers import BertTokenizer
from transfomers import BertForSequenceClassification

import huggingface_hub
import data, utils
import models
parser=argparse.ArgumentParser()
parser.add_argument('--max-len',type=int,default=128)
parser.add_argument('--max-len-vader',type=int,default=40)
parser.add_argument('--batch-size',type=int,default=32)
parser.add_argument('--epochs',default=5,type=int)
parser.add_argument('--use-vader',default=False,type=bool)
parser.add_argument('--use-bert',default=False,type=bool)
parser.add_argument('--use-cnn',default=False,type=bool)

args=parser.parse_args()

DATA_FOLDER = Path("starter")
OUT_FOLDER = Path("models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
list_to_device = lambda th_obj: [tensor.to(device) for tensor in th_obj]


# Higher bound settings: MAX_LEN = 256 and BATCH_SIZE = 16
yelp_reviews = data.load_json(DATA_FOLDER / "yelp_review_training_dataset.jsonl")
print("loaded", len(yelp_reviews), "data points")


if args.use_bert:
    xlnet_tokenizer = torch.hub.load(
    "huggingface/pytorch-transformers", "tokenizer", "xlnet-base-cased"
    )
    model_tokenizer = torch.hub.load(
        "huggingface/pytorch-transformers", "tokenizer", "xlnet-base-cased"
    )
# tokenize_review(model_tokenizer, "I love this grub!")

# train 75% | validation 15% | test 10%
train_ratio = 0.75
validate_ratio = 0.15
test_ratio = 0.10
assert train_ratio + validate_ratio + test_ratio == 1
train_reviews, validate_reviews, test_reviews = train_validate_test_split(
    yelp_reviews, train_ratio, validate_ratio
)
# train_reviews_df, val_reviews_df, test_reviews_df = train_validate_test_split(yelp_reviews, train_ratio, validate_ratio)
# train_reviews, train_reviews_target, train_reviews_mask = format_reviews(xlnet_tokenizer, train_reviews_df)
# validate_reviews, test_reviews_target, validate_reviews_mask = format_reviews(xlnet_tokenizer, validate_reviews_df)
# test_reviews, test_reviews_target, _ = format_reviews(xlnet_tokenizer, test_reviews_df)
print(len(train_reviews.index), "yelp reviews for training")
train_reviews
print(len(validate_reviews.index), "yelp reviews for validation")
validate_reviews
print(len(test_reviews.index), "yelp reviews for testing")
test_reviews
model = LanguageModel(vocab_size=MAX_LEN, rnn_size=256, vader_size=MAX_LEN_VADER)
# set model to training mode
model.train()
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# TODO: fix this block
DATASET = train_reviews


since = time.time()

# start training
for epoch in range(num_epochs):
    indices = np.random.permutation(range(DATASET.size))
    t = tqdm.notebook.tqdm(range(0, (DATASET.size // batch_size) + 1))

    for i in t:
        # batch
        batch = format_reviews(
            xlnet_tokenizer, DATASET, indices[i * batch_size : (i + 1) * batch_size]
        )
        (batch_input, batch_target, batch_target_mask) = batch_to_torch(*batch)
        for item in (batch_input, batch_target, batch_target_mask):
            print(item.size())
        (batch_input, batch_target, batch_target_mask) = list_to_device(
            (batch_input, batch_target, batch_target_mask)
        )

        # forward pass
        prediction = model.forward(batch_input)
        loss = loss_fn(prediction, batch_target, batch_target_mask)
        losses.append(loss.item())
        accuracy = (
            th.eq(prediction.argmax(dim=2, keepdim=False), batch_target).float()
            * batch_target_mask
        ).sum() / batch_target_mask.sum()
        accuracies.append(accuracy.item())

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # visuallize data
        if i % 100 == 0:
            batch_val = build_batch(d_valid, range(len(d_valid)))
            (batch_input_val, batch_target_val, batch_target_mask_val) = list_to_device(
                batch_to_torch(*batch_val)
            )
            prediction_val = model.forward(batch_input_val)
            loss_val = loss_fn(prediction_val, batch_target_val, batch_target_mask_val)
            print("Evaluation set loss:", loss_val.item())
            print(
                f"Epoch: {epoch} Iteration: {i} Loss: {np.mean(losses[-10:])} Accuracy: {np.mean(accuracies[-10:])}"
            )

# set model to evaluation model
model.eval()
