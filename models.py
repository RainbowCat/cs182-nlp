import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import (
    LSTM,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Linear,
    MaxPool2d,
    Sequential,
)
from torchmetrics import Accuracy, MetricCollection, StatScores


class LanguageModel(pl.LightningModule):
    def __init__(
        self,
        args,
        num_layers: int = 1,
        dropout: float = 0,
    ):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        metrics = MetricCollection([Accuracy(), StatScores()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.base_model = torch.hub.load(
            "huggingface/pytorch-transformers",
            "model",
            "bert-base-cased" if self.args.use_bert else "xlnet-base-cased",
        )

        # freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        H, W = 5, 5  # along Embedding Length

        self.conv = Conv2d(1, 1, (H, W))
        self.mp = MaxPool2d((self.args.max_len - ((H - 1) // 2) * 2, 1))

        if self.args.use_vader:
            self.vader_size = self.args.max_len_vader
        else:
            self.vader_size = 0

        self.lstm = LSTM(
            input_size=1,
            hidden_size=1,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.dropout = Dropout(dropout)

        self.dense = Linear(768 - ((W - 1) // 2) * 2 // 1 + self.vader_size, 100)
        # classify yelp_reviews into 5 ratings
        self.output = Linear(100, 5)

    def forward(self, encodings, sentiments):
        out = self.base_model(**encodings)

        out_hidden = out.last_hidden_state
        batches_len, word_len, embedding_len = out_hidden.shape
        out_hidden = out_hidden.reshape(batches_len, 1, word_len, embedding_len)
        result = self.mp(self.conv(out_hidden))
        input1 = result.squeeze(1).squeeze(1)

        if self.args.use_vader:
            batch_size, vader_len = sentiments.shape
            output, _ = self.lstm(sentiments.reshape(batch_size, vader_len, 1))
            input2 = output.squeeze(2)
            combined_input = (input1, input2)
        else:
            combined_input = (input1,)  # Tuples need the stray comma

        combined_input = torch.cat(combined_input, dim=1)

        lstm_drop = self.dropout(combined_input)

        logits = F.relu(self.dense(lstm_drop))
        logits = self.output(logits)
        return logits

    def loss_fn(self, prediction, target):
        loss_criterion = CrossEntropyLoss(reduction="none")
        return torch.mean(loss_criterion(prediction, target))

    def training_step(self, batch, batch_idx):
        encoding, sentiment, target = batch
        prediction = self(encoding, sentiment)
        loss = self.loss_fn(prediction, target)
        self.log("train_loss", loss)
        self.log_dict(self.train_metrics(prediction, target))
        return loss

    def validation_step(self, batch, batch_idx):
        encoding, sentiment, target = batch
        prediction = self(encoding, sentiment)
        loss = self.loss_fn(prediction, target)
        self.log("val_loss", loss)
        self.log_dict(self.val_metrics(prediction, target))
        return loss

    def test_step(self, batch, batch_idx):
        encoding, sentiment, target = batch
        prediction = self(encoding, sentiment)
        loss = self.loss_fn(prediction, target)
        self.log("test_loss", loss)
        self.log_dict(self.test_metrics(prediction, target))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
