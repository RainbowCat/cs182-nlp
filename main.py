import itertools
from argparse import Namespace

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

import data
import models

for use_bert, use_vader in itertools.product([True, False], repeat=2):
    print(f"{use_bert, use_vader=}")

    # try:
    args = Namespace(
        # TODO add back
        # batch_size=240,
        batch_size=3,
        # TODO add back
        # epochs=10,
        epochs=1,
        max_len=128,
        max_len_vader=128,
        use_bert=use_bert,
        use_cnn=True,
        use_vader=use_vader,
    )

    model = models.LanguageModel(args)
    datamodule = data.YelpDataModule(args)
    trainer = pl.Trainer(
        # TODO add back
        # gpus=torch.cuda.device_count(),
        gpus=0,
        max_epochs=args.epochs,
        # overfit_batches=1,
        # track_grad_norm=2,
        weights_summary="full",
        progress_bar_refresh_rate=1,
        check_val_every_n_epoch=1,
        callbacks=[ModelCheckpoint(monitor="val_loss", every_n_train_steps=100)],
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
# except Exception as e:
# print(e)
# continue
