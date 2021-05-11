import copy
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import huggingface_hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import data
import models

DATA_FOLDER = Path("starter")
from argparse import Namespace

args = Namespace(
    batch_size=32,
    epochs=10,
    max_len=128,
    max_len_vader=40,
    use_bert=False,
    use_cnn=False,
    use_vader=False,
)
try:
    yelp_dataset = pickle.load("datamodule.pkl")
except:
    yelp_dataset = data.YelpDataModule(
        args, data_path="starter/yelp_review_training_dataset.jsonl"
    )
    with open("datamodule.pkl", "wb") as f:
        pickle.dump(yelp_dataset, f)

model = models.LanguageModel(args)
trainer = pl.Trainer(
    gpus=torch.cuda.device_count(),
    # overfit_batches=1,
    # track_grad_norm=2,
    weights_summary="full",
    progress_bar_refresh_rate=100,
    check_val_every_n_epoch=5,
)
trainer.fit(model, datamodule=yelp_dataset)
trainer.validate(model, datamodule=yelp_dataset)
