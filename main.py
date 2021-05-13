import argparse
import itertools
from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import data
import models


def main(
    use_bert: bool,
    use_vader: bool,
    use_cnn: bool,
    batch_size: int = 256,
    epochs: int = 10,
    max_len: int = 128,
    max_len_vader: int = 128,
    gpus=0,
) -> None:
    args = Namespace(
        batch_size=batch_size,
        epochs=epochs,
        max_len=max_len,
        max_len_vader=max_len_vader,
        use_bert=use_bert,
        use_vader=use_vader,
        use_cnn=use_cnn,
    )

    model = models.LanguageModel(args)
    datamodule = data.YelpDataModule(args)
    trainer = pl.Trainer(
        auto_select_gpus=True,
        gpus=[gpus],
        max_epochs=args.epochs,
        # overfit_batches=1,
        # track_grad_norm=2,
        weights_summary="full",
        progress_bar_refresh_rate=1,
        check_val_every_n_epoch=2,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint(f"bert={use_bert}+cnn={use_cnn}+vader={use_vader}.ckpt")
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-bert", type=eval)
    parser.add_argument("--use-vader", type=bool)
    parser.add_argument("--use-cnn", type=bool)
    parser.add_argument("--gpu-id", type=int)
    args = parser.parse_args()

    for use_bert, use_cnn, use_vader in itertools.product([False, True], repeat=3):
        if Path(f"bert={use_bert}+cnn={use_cnn}+vader={use_vader}.ckpt").exists():
            continue

        # main(
        #     use_bert=True, use_cnn=False, use_vader=False,
        #     epochs=2,
        #     batch_size=2048,
        # )
        # main(
        #     use_bert=True, use_cnn=False, use_vader=True,
        #     epochs=2,
        #     batch_size=2048,
        # )
        # main(
        #     use_bert=True, use_cnn=True, use_vader=True,
        #     epochs=2,
        #     batch_size=2048,
        # )
        # main(
        #     use_bert=True, use_cnn=True, use_vader=True,
        #     epochs=2,
        #     batch_size=2048,
        # )
        # main(
        #     use_bert=True, use_cnn=True, use_vader=True,
        #     epochs=2,
        #     batch_size=2048,
        # )

        main(
            use_bert=use_bert,
            use_vader=use_vader,
            use_cnn=use_cnn,
            epochs=2,
            batch_size=2048 if args.use_bert else 512,
        )
