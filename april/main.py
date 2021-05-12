import itertools
from argparse import Namespace

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import data
import models


def main(
    use_bert: bool,
    use_vader: bool,
    batch_size: int = 240,
    epochs: int = 10,
    max_len: int = 128,
    max_len_vader: int = 128,
) -> None:
    args = Namespace(
        batch_size=batch_size,
        epochs=epochs,
        max_len=max_len,
        max_len_vader=max_len_vader,
        use_bert=use_bert,
        use_vader=use_vader,
    )

    model = models.LanguageModel(args)
    datamodule = data.YelpDataModule(args)
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.epochs,
        # overfit_batches=1,
        # track_grad_norm=2,
        weights_summary="full",
        progress_bar_refresh_rate=1,
        check_val_every_n_epoch=1,
        callbacks=[ModelCheckpoint(monitor="val_loss", every_n_train_steps=1_000)],
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    for use_bert, use_vader in itertools.product([True, False], repeat=2):
        print(f"{use_bert, use_vader=}")
        main(use_bert, use_vader)
