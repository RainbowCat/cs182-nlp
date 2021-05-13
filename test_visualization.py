import itertools
import json
import os
import sys
from pathlib import Path
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix

import data
import models


def mae_accuracy(dataset, model_name, prefix) -> None:
    encoding, sentiment = dataset.yelp_reviews[0], dataset.yelp_reviews[1]
    # add 1 to turn into proper 5 star ratings
    preds = (model(encoding, sentiment).argmax(1) + 1).float().tolist()
    full_table = dataset.df
    full_table["pred"] = preds
    full_table["dist"] = np.abs(full_table["pred"] - full_table["stars"])

    Path(model_name).with_suffix(f".txt{prefix}+{randrange(50)}").write_text(
        str(
            {
                f"mae_{prefix}": np.sum(full_table["dist"]) / 500,
                f"acc_{prefix}": len(full_table[full_table["dist"] == 0]) / 500,
            }
        )
    )


def matrix(dataset1, dataset2, model_name: str) -> None:
    encoding1, sentiment1 = dataset1.yelp_reviews[0], dataset1.yelp_reviews[1]
    # add 1 to turn into proper 5 star ratings
    preds1 = (model(encoding1, sentiment1).argmax(1) + 1).float().tolist()

    encoding2, sentiment2 = dataset2.yelp_reviews[0], dataset2.yelp_reviews[1]
    # add 1 to turn into proper 5 star ratings
    preds2 = (model(encoding1, sentiment1).argmax(1) + 1).float().tolist()

    dataset1.df["pred"] = preds1
    dataset2.df["pred"] = preds2

    full_table = pandas.concat([dataset1.df, dataset2.df])

    confusion_mtx = confusion_matrix(full_table["stars"], full_table["pred"])
    print(confusion_mtx)
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes=range(1, 5), model_name=model_name)
    plt.show()


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    model_name="dummy",
):
    """This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    fig, ax = plt.subplots()
    im = ax.matshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im)
    # tick_marks = np.arange(len(classes))
    # ax.set_xticklabels(tick_marks, classes, rotation=45)
    # ax.set_yticklabels(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    fig.savefig(f"matrix-{randrange(50)}-{model_name.with_suffix('.png')}")

    # with Path("out_tmp.json").open("w") as f:
    # for line in d:
    # print(json.dumps(line), file=f)


if __name__ == "__main__":
    for model_name in [
        Path("fff.ckpt"),
        Path("fft.ckpt"),
        Path("ftf.ckpt"),
        Path("ftt.ckpt"),
        Path("tff.ckpt"),
        Path("tft.ckpt"),
        Path("ttf.ckpt"),
        Path("ttt.ckpt"),
    ]:
        model = models.LanguageModel.load_from_checkpoint(
            model_name,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        model.eval()
        dataset_5 = data.YelpDataset(model.args, sys.argv[1])
        dataset_8 = data.YelpDataset(model.args, sys.argv[2])

        mae_accuracy(dataset_5, model_name, 5)
        mae_accuracy(dataset_8, model_name, 8)
        matrix(dataset_5, dataset_8, model_name)
