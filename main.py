#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from itertools import chain, product
from os import PathLike
from pathlib import Path
from typing import (Dict, Iterable, List, Mapping, NamedTuple, Optional,
                    Sequence)

import huggingface_hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from segtok import tokenizer
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from transformers import BertForSequenceClassification, BertTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import data
import models

parser = argparse.ArgumentParser()
parser.add_argument("--max-len", type=int, default=128)
parser.add_argument("--max-len-vader", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--use-vader", default=False, type=bool)
parser.add_argument("--use-bert", default=False, type=bool)
parser.add_argument("--use-cnn", default=False, type=bool)

args = parser.parse_args()
print("=== SETUP ===")
print(args)

DATA_FOLDER = Path("starter")
OUT_FOLDER = Path("models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on " + str(device))
list_to_device = lambda th_obj: [tensor.to(device) for tensor in th_obj]


print("=== DATASET ===")
yelp_reviews_df = data.load_json(DATA_FOLDER / "yelp_review_training_dataset.jsonl")
print("loaded", len(yelp_reviews_df), "data points")
yelp_reviews = data.format_reviews(args, tokenizer, yelp_reviews_df)
print("formatted", len(yelp_reviews_df), "reviews")

# train 75% | validation 15% | test 10%
train_ratio = 0.75
validate_ratio = 0.15
test_ratio = 0.10
assert train_ratio + validate_ratio + test_ratio == 1.0
train_reviews, validate_reviews, test_reviews = data.train_validate_test_split(
    yelp_reviews, train_ratio, validate_ratio
)


def test_eval(args, model, use_all=False, mode="val"):
    reviews_dataset = None
    if mode == "val":
        print("Running Validation")
        mode = "Validation"
        reviews_dataset = validate_reviews
    elif mode == "test":
        print("Running Testing")
        mode = "Test"
        reviews_dataset = test_reviews
    else:
        assert False, "Invalid mode"
    num_of_review_set = len(reviews_dataset) if use_all else 1000
    indices = np.random.permutation(len(reviews_dataset))
    num_batches = tqdm.tqdm(
        range(
            0,
            (num_of_review_set // args.batch_size)
            + (1 if num_of_review_set % args.batch_size > 0 else 0),
        )
    )
    loss_val_total = 0
    accuracy_val_total = 0
    temp_count = 0
    for i in num_batches:
        val_start_i = i * args.batch_size
        val_end_i = (i + 1) * args.batch_size
        # print(val_start_i, val_end_i, indices.shape)
        (
            batch_input_val,
            batch_target_val,
            batch_target_mask_val,
            batch_review_sentiment_val,
        ) = data.format_reviews(
            args,
            model.tokenizer,
            reviews_dataset,
            indices[val_start_i:val_end_i],
        )
        # print(batch_input_val.shape, batch_review_sentiment_val.shape)
        (batch_input_val, batch_target_val) = list_to_device(
            (batch_input_val, batch_target_val)
        )
        batch_target_mask_val, batch_review_sentiment_val = list_to_device(
            (batch_target_mask_val, batch_review_sentiment_val)
        )
        # print(batch_input_val.shape, batch_review_sentiment_val.shape)
        prediction_val = model.forward(batch_input_val, batch_review_sentiment_val)
        # print(prediction_val.size(), batch_target_val.size())
        # print(prediction_val, batch_target_val)
        loss_val_total += model.loss_fn(prediction_val, batch_target_val).item()
        # print(loss_val)
        accuracy_val_total += torch.mean(
            torch.eq(
                prediction_val.argmax(dim=1, keepdim=False), batch_target_val
            ).float()
        ).item()
        temp_count += 1
        if i % round(8000 / args.batch_size) == 0 and i != 0 and use_all:
            print(
                mode,
                "Prelim Evaluation set loss:",
                loss_val_total / temp_count,
                mode,
                "Prelim Accuracy:",
                accuracy_val_total / temp_count,
            )
    loss_val = loss_val_total / temp_count
    accuracy_val = accuracy_val_total / temp_count
    print(mode, "Evaluation set loss:", loss_val, mode, "Accuracy set %:", accuracy_val)


print("creating model")
model = models.LanguageModel(
    vocab_size=args.max_len,
    vader_size=args.max_len_vader,
    use_vader=args.use_vader,
    use_bert=args.use_bert,
    use_cnn=args.use_cnn,
)

# start training
print("=== TRAINING ===")
model.train()

since = time.time()
losses = []
accuracies = []
t_start = 0

lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
model = model.to(device)

print(f"params: {args.epochs=}, {lr=}")
for epoch in range(args.epochs):
    indices = np.random.permutation(train_reviews.shape[0])

    dataset_batch_cap = (train_reviews.shape[0] // args.batch_size) + (
        1 if train_reviews.shape[0] % args.batch_size > 0 else 0
    )

    for i in tqdm.trange(dataset_batch_cap):
        # batching
        (
            batch_input,
            batch_target,
            batch_target_mask,
            batch_review_sentiments,
        ) = data.format_reviews(
            args,
            model.tokenizer,
            train_reviews,
            indices[i * args.batch_size : (i + 1) * args.batch_size],
        )
        (
            batch_input,
            batch_target,
            batch_target_mask,
            batch_review_sentiments,
        ) = list_to_device(
            (batch_input, batch_target, batch_target_mask, batch_review_sentiments)
        )

        # forward pass
        # print(batch_input.shape, batch_review_sentiments.shape)
        prediction = model(batch_input, batch_review_sentiments)
        loss = model.loss_fn(prediction, batch_target)
        losses.append(loss.item())
        accuracy = torch.mean(
            torch.eq(prediction.argmax(dim=1, keepdim=False), batch_target).float()
        )
        accuracies.append(accuracy.item())

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # visualize data
        if i % 1000 == 0 and i != t_start:
            models.save_model(
                args, OUT_FOLDER / str(time.time()), model, losses, accuracies
            )
            models.test_eval(args, model)
            print(
                f"Epoch: {epoch} Iteration: {i} Train Loss: {np.mean(losses[-10:])} Train Accuracy: {np.mean(accuracies[-10:])}"
            )

print("=== VALIDATION ===")
# set model to evaluation model
model.eval()
