import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

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
    return padded


def encode_reviews(args, df: pd.DataFrame, stage: str = "train"):
    tokenizer = torch.hub.load(
        "huggingface/pytorch-transformers",
        "tokenizer",
        "bert-base-cased" if args.use_bert else "xlnet-base-cased",
    )

    now = time.time()
    encoding = tokenizer(
        df.text.to_list(),
        add_special_tokens=True,
        max_length=args.max_len,
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    print(f"Tokenization took {time.time()-now} seconds")

    analyzer = SentimentIntensityAnalyzer()
    sentiment = torch.tensor(
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
    # We subtract 1 to get valid range [0, N).
    target = torch.tensor(df.stars - 1) if stage != "test" else None

    return (
        encoding,
        sentiment,
        target,
    )


class YelpDataset(Dataset):
    def __init__(self, args, data_path):
        super().__init__()
        self.data_path = data_path

        self.df = pd.read_json(self.data_path, orient="records", lines=True)
        self.len = len(self.df)
        self.yelp_reviews = encode_reviews(args, self.df)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        encodings, sentiments, target = self.yelp_reviews
        # add targets only if they exist
        return (
            {k: v[idx] for k, v in encodings.items()},
            sentiments[idx],
            target[idx] if target is not None else None,
        )


class YelpDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        data_path: os.PathLike = Path("starter/yelp_review_training_dataset.jsonl"),
    ):
        super().__init__()
        self.args = args
        self.data_path = data_path

    def setup(self, stage: Optional[str] = None):

        if stage == "test":
            self.dataset = YelpDataset(self.args, self.data_path)
        else:
            PRECOMPUTED_DATA = Path(
                "bert-tokens.pkl" if self.args.use_bert else "xlnet-tokens.pkl"
            )

            try:
                with open(PRECOMPUTED_DATA, "rb") as f:
                    self.dataset = pickle.load(f)
            except FileNotFoundError:
                self.dataset = YelpDataset(self.args, self.data_path)
                with open(PRECOMPUTED_DATA, "wb") as f:
                    pickle.dump(self.dataset, f)

        N = len(self.dataset)
        num_train = int(0.9 * N) if stage != "test" else 0
        num_val = int(0.05 * N) if stage != "test" else 0
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
            num_workers=int(os.cpu_count() / 16),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=int(os.cpu_count() / 16),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=int(os.cpu_count() / 16),
        )
