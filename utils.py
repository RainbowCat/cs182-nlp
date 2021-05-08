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
from segtok import tokenizer
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MAX_LEN = 128
MAX_LEN_VADER = 40
BATCH_SIZE = 32
EPOCHS = 5

# Higher bound settings: MAX_LEN = 256 and BATCH_SIZE = 16

list_to_device = lambda th_obj: [tensor.to(device) for tensor in th_obj]
