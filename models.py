import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import pytorch_lightning as pl
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

import data
from utils import *


class LanguageModel(pl.LightningModule):
    def __init__(
        self,
        args,
        num_layers: int = 1,
        dropout: float = 0,
    ):
        super().__init__()

        self.use_vader = args.use_vader
        self.use_bert = args.use_bert
        self.use_cnn = args.use_cnn

        self.model_name = "bert-base-cased" if args.use_bert else "xlnet-base-cased"
        self.tokenizer = torch.hub.load(
            "huggingface/pytorch-transformers", "tokenizer", self.model_name
        )

        self.base_model = torch.hub.load(
            "huggingface/pytorch-transformers", "model", self.model_name
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
        conv2d_out_Hout = args.max_len - ((conv2d_kernel_H - 1) // 2) * 2  # Vocab Size
        conv2d_out_Wout = 768 - ((conv2d_kernel_W - 1) // 2) * 2  # length

        self.max_pool_2d = nn.MaxPool2d((conv2d_out_Hout, 1))
        max_pool_2d_out_height = conv2d_out_Hout // conv2d_out_Hout
        max_pool_2d_out_length = conv2d_out_Wout // 1

        self.lstm = None
        self.vader_size = 0
        # self.sentiments = {}
        if self.use_vader:
            self.vader_size = args.max_len_vader
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=1,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )

        self.dropout = nn.Dropout(dropout)
        # print(max_pool_2d_out_length + self.vader_size)

        hidden_layer_dense = 100

        self.dense = nn.Sequential(
            nn.Linear(max_pool_2d_out_length + self.vader_size, hidden_layer_dense),
            nn.ReLU(),
        )
        print(max_pool_2d_out_height + self.vader_size, hidden_layer_dense)
        self.output = nn.Linear(
            hidden_layer_dense, 5
        )  # classify yelp_reviews into 5 ratings

    def training_step(self, batch, batch_idx):
        (
            batch_input,
            batch_target,
            batch_review_sentiments,
            batch_target_mask,
        ) = batch
        (
            batch_input,
            batch_target,
            batch_target_mask,
            batch_review_sentiments,
        ) = list_to_device(
            (batch_input, batch_target, batch_target_mask, batch_review_sentiments)
        )
        prediction = self.forward(batch_input, batch_review_sentiments)
        loss = self.loss_fn(prediction, batch_target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, words, sentiments):
        out = self.base_model(words)
        out_hidden = out.last_hidden_state
        batches_len, word_len, embedding_len = out_hidden.shape
        out_hidden = out_hidden.reshape(batches_len, 1, word_len, embedding_len)
        conv2d_out = self.conv2D_layer(out_hidden)
        result = self.max_pool_2d(conv2d_out)
        input1 = result.squeeze(1).squeeze(1)

        if self.use_vader and self.lstm:
            batch_size, vader_len = sentiments.shape
            output, _ = self.lstm(sentiments.reshape(batch_size, vader_len, 1))
            input2 = output.squeeze(2)
            combined_input = (input1, input2)
        else:
            combined_input = (input1,)  # Tuples need the stray comma

        combined_input = torch.cat(combined_input, dim=1)

        lstm_drop = self.dropout(combined_input)
        print("dropped")

        conv2d_c_in = 1
        conv2d_c_out = 1
        conv2d_kernel_W = 5  # along Embedding Length
        conv2d_kernel_H = 5  # along Word Length
        conv2d_out_Hout = (
            args.max_len - ((conv2d_kernel_H - 1) // 2) * 2
        )  # Vocab Size
        conv2d_out_Wout = 768 - ((conv2d_kernel_W - 1) // 2) * 2  # length

        self.max_pool_2d = nn.MaxPool2d((conv2d_out_Hout, 1))
        max_pool_2d_out_height = conv2d_out_Hout // conv2d_out_Hout
        max_pool_2d_out_length = conv2d_out_Wout // 1

        logits = nn.Linear(max_pool_2d_out_length + args.max_len_vader, 100)(lstm_drop)
        print("hi")
        logits = nn.ReLU()(logits)
        # logits = self.dense(lstm_drop)
        # print("logits")
        logits = self.output(logits)
        print("logits 2", logits.shape)
        return logits

    def predict(self, vectorized_words, vadar_sentiments):
        logits = self.forward(vectorized_words, vadar_sentiments)
        prediction = logits.argmax(dim=1, keepdim=False)
        print(prediction)
        return prediction

    def loss_fn(self, prediction, target):
        loss_criterion = nn.CrossEntropyLoss(reduction="none")
        # print(prediction.shape,target.shape)
        return torch.mean(loss_criterion(prediction, target))


def save_model(args, filename, model, losses, accuracies):
    torch.save(
        {
            "args": args,
            "state_dict": model.state_dict(),
            "losses": losses,
            "accuracies": accuracies,
        },
        filename,
    )
