import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import nltk
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# from torchvision import datasets, models, transforms
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("punkt")


def pad_sequence(numerized, pad_index, to_length, beginning=False):
    pad = numerized[:to_length]
    if beginning:
        padded = [pad_index] * (to_length - len(pad)) + pad
    else:
        padded = pad + [pad_index] * (to_length - len(pad))
    # mask = [w != pad_index for w in padded]
    return padded


def encode_reviews(args, df: pd.DataFrame):
    tokenizer = torch.hub.load(
        "huggingface/pytorch-transformers",
        "tokenizer",
        "bert-base-cased" if args.use_bert else "xlnet-base-cased",
    )

    now = time.time()
    encodings = tokenizer(
        df.text.to_list(),
        add_special_tokens=True,
        max_length=args.max_len,
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )

    print(f"Tokenization took {time.time()-now} seconds")

    analyzer = SentimentIntensityAnalyzer()
    sentiments = torch.tensor(
        [
            pad_sequence(
                [
                    analyzer.polarity_scores(s)["compound"]
                    for s in nltk.tokenize.sent_tokenize(text)
                ],
                pad_index=0,
                to_length=args.max_len_vader,
            )
            for text in tqdm(df.text)
        ]
    )
    # we subtract 1 to get valid range [0,N)
    targets = torch.tensor(df.stars - 1)

    return (
        encodings,
        sentiments,
        targets,
    )


class YelpDataset(Dataset):
    def __init__(self, args, data_path):
        super().__init__()
        self.data_path = data_path

        yelp_reviews_df = pd.read_json(self.data_path, orient="records", lines=True)
        self.len = len(yelp_reviews_df)
        self.yelp_reviews = encode_reviews(args, yelp_reviews_df)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        encodings, sentiments, target = self.yelp_reviews
        return {k: v[idx] for k, v in encodings.items()}, sentiments[idx], target[idx]


def collate(batch):
    # list[(...)] -> ([]..)
    print(f"{batch[0][0]['input_ids'].shape=}")
    sys.exit()


class YelpDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        data_path: os.PathLike = Path("starter/yelp_review_training_dataset.jsonl"),
    ):
        super().__init__()
        self.args = args

        PRECOMPUTED_DATA = Path(
            "bert-tokens.pkl" if args.use_bert else "xlnet-tokens.pkl"
        )

        try:
            with open(PRECOMPUTED_DATA, "rb") as f:
                self.dataset = pickle.load(f)
        except FileNotFoundError:
            self.dataset = YelpDataset(args, data_path)
            with open(PRECOMPUTED_DATA, "wb") as f:
                pickle.dump(self.dataset, f)

    def setup(self, stage: Optional[str] = None):

        N = len(self.dataset)
        num_train = int(0.6 * N)
        num_val = int(0.2 * N)
        num_test = N - num_train - num_val

        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, [num_train, num_val, num_test]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count() // 2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count() // 4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count() // 4,
        )
