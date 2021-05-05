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

MAX_LEN = 256

class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        rnn_size,
        embedding_layer,
        num_layers=1,
        dropout=0,
        max_len=MAX_LEN,
    ):
        super().__init__()

        #############
        #  INPUT 1  #
        #############
        # Create an embedding layer of shape [vocab_size, rnn_size]
        # Use nn.Embedding
        # That will map each word in our vocab into a vector of rnn_size size.
        self.xlnet = torch.hub.load(
            "huggingface/pytorch-transformers", "model", "xlnet-base-cased"
        )
        # Output: (max_len x 768), where 768 hidden layers of XLNet
        #################
        #  INPUT 1 END  #
        #################

        #############
        #  INPUT 2  #
        #############
        self.analyzer = SentimentIntensityAnalyzer()
        self.lstm = nn.LSTM(
            input_size=rnn_size,
            hidden_size=rnn_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # Use Embedding results directly
        #################
        #  INPUT 2 END  #
        #################

        # Coming in: torch.Size([BATCH_SIZE, MAX_LEN, 768])
        #   (XLNet has 768 hidden layers, https://huggingface.co/transformers/pretrained_models.html)
        conv2d_width = 5
        conv2d_height = 5

        self.conv2D_layer = nn.Conv2D(
            1, 1, (conv2d_width, conv2d_height)
        )  # Filter of 5 x 5, Cin = 1, Cout = 1

        # Output: (768 - ((CONV2D_width - 1) / 2) * 2) by (MAX_LEN - ((CONV2D_height - 1) / 2) * 2)
        conv2d_out_height = max_len - ((conv2d_height - 1) // 2) * 2
        conv2d_out_length = 768 - ((conv2d_width - 1) // 2) * 2

        self.max_pool_2d = nn.MaxPool2d((2, 2))
        max_pool_2d_out_height = conv2d_out_height // 2
        max_pool_2d_out_length = conv2d_out_length // 2

        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Sequential(nn.Linear(rnn_size, vocab_size), nn.ReLu())
        self.output = nn.Linear(
            rnn_size, 10
        )  # classify yelp_reviews into 10 rating levels

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        lstm_drop = self.dropout(lstm_out)
        logits = self.dense(lstm_drop)
        logits = self.output(logits)
        return logits

    def loss_fn(self, prediction, target, mask):
        if classes is None:
            raise NotImplementedError
        else:
            # Regression
            pass
