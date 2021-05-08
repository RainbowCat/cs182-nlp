import copy
import nltk
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
from numpy.lib.function_base import vectorize
from segtok import tokenizer
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from transformers import BertForSequenceClassification, BertTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import *


class LanguageModel(nn.Module):
    def __init__(
        self,
        args,
        use_vader,
        use_bert,
        use_cnn,
        vocab_size,
        rnn_size,
        vader_size,
        num_layers=1,
        dropout=0,
    ):
        super().__init__()

        # Create an embedding layer, with 768 hidden layers
        if use_bert:
            self.base_model = BertForSequenceClassification.from_pretrained(
                "bert-base-cased", num_labels=200
            )
            self.base_model.classifier.add_module("bert_activation", nn.Tanh())
            self.base_model.classifier.add_module("prediction", nn.Linear(200, 5))
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        else:
            self.base_model = torch.hub.load(
                "huggingface/pytorch-transformers", "model", "xlnet-base-cased"
            )
            self.tokenizer = torch.hub.load(
                "huggingface/pytorch-transformers", "tokenizer", "xlnet-base-cased"
            )

        for param in self.base_model.layer.parameters():
            param.requires_grad = False
        # Output: (vocab_size x 768), where 768 hidden layers of base_model

        # Coming in: torch.Size([BATCH_SIZE, vocab_size, 768])
        #   (XLNet has 768 hidden layers, https://huggingface.co/transformers/pretrained_models.html)
        conv2d_c_in = 1
        conv2d_c_out = 1
        conv2d_kernel_W = 5  # along Embedding Length
        conv2d_kernel_H = 5  # along Word Length

        self.conv2D_layer = nn.Conv2d(
            conv2d_c_in, conv2d_c_out, (conv2d_kernel_H, conv2d_kernel_W)
        )
        # Filter of (conv2d_kernel_H, conv2d_kernel_W), Cin = 1, Cout = 1

        # Output:
        conv2d_out_Hout = vocab_size - ((conv2d_kernel_H - 1) // 2) * 2  # Vocab Size
        conv2d_out_Wout = 768 - ((conv2d_kernel_W - 1) // 2) * 2  # length

        self.max_pool_2d = nn.MaxPool2d((conv2d_out_Hout, 1))
        max_pool_2d_out_height = conv2d_out_Hout // conv2d_out_Hout
        max_pool_2d_out_length = conv2d_out_Wout // 1

        self.lstm = None
        self.sentiments = {}
        if use_vader:
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=1,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            # Create dictionary of all the reviews' Vader temporarily
            review_iterator = tqdm.notebook.tqdm(
                yelp_reviews.iterrows(), total=yelp_reviews.shape[0]
            )

            for i, review in review_iterator:
                # Tokenize by TOKENIZER
                review_text = review["text"]
                # VADER
                sentence_list = nltk.tokenize.sent_tokenize(review_text)
                review_sentiment_sentence = []
                analyzer= SentimentIntensityAnalyzer()
                for sentence in sentence_list:
                    vs = analyzer.polarity_scores(sentence)
                    review_sentiment_sentence.append(vs["compound"])
                # TODO should last arg be self.vader_size?
                padded, _ = data.pad_sequence(review_sentiment_sentence, 0, vocab_size)
                self.sentiments[review["review_id"]] = padded
                if len(self.sentiments) < 20:
                    print(len(self.sentiments), self.sentiments[review["review_id"]])
        else:
            vader_size = 0

        self.dropout = nn.Dropout(dropout)
        # print(max_pool_2d_out_length + vader_size)

        hidden_layer_dense = 100

        self.dense = nn.Sequential(
            nn.Linear(max_pool_2d_out_length + vader_size, hidden_layer_dense),
            nn.ReLU(),
        )
        self.output = nn.Linear(
            hidden_layer_dense, 5
        )  # classify yelp_reviews into 5 ratings

    def forward(self, vectorized_words, vader):
        xlnet_out = self.base_model(vectorized_words)
        xlnet_out_hidden = xlnet_out.last_hidden_state
        batches_len, word_len, embedding_len = xlnet_out_hidden.shape
        xlnet_out_hidden = xlnet_out_hidden.reshape(
            batches_len, 1, word_len, embedding_len
        )
        conv2d_out = self.conv2D_layer(xlnet_out_hidden)
        result = self.max_pool_2d(conv2d_out)
        # print(result.shape)
        input1 = result.squeeze(1).squeeze(1)

        if self.lstm:
            batch_size, vader_len = vader.shape
            # print(x.reshape(batch_size, vader_len, 1).shape)
            output, _ = self.lstm(vader.reshape(batch_size, vader_len, 1))
            # print(output.shape)
            input2 = output.squeeze(2)
            combined_input = (input1, input2)
        else:
            combined_input = (input1,)  # Tuples need the stray comma

        combined_input = torch.cat(combined_input, dim=1)

        lstm_drop = self.dropout(combined_input)
        logits = self.dense(lstm_drop)
        logits = self.output(logits)
        return logits

    def loss_fn(self, prediction, target):
        loss_criterion = nn.CrossEntropyLoss(reduction="none")
        print(prediction.shape,target.shape)
        return torch.mean(loss_criterion(prediction, target))
