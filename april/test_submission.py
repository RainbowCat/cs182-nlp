import json
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

import data
import models


def test_model(path: os.PathLike) -> None:
    path=Path(path)
    with torch.no_grad():

        model = models.LanguageModel.load_from_checkpoint(
            "lightning_logs/version_67/checkpoints/epoch=3-step=4999.ckpt",
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        model.eval()
        dataset = data.YelpDataset(model.args, path)
        encoding, sentiment = dataset.yelp_reviews[0], dataset.yelp_reviews[1]
        # add 1 to turn into proper 5 star ratings
        preds = (model(encoding, sentiment).argmax(1) + 1).float().tolist()
        d = [
            {"review_id": id, "predicted_stars": s}
            for id, s in zip(dataset.df.review_id, preds)
        ]

    with Path("out_tmp.json").open("w") as f:
        for line in d:
            print(json.dumps(line), file=f)

if __name__ == "__main__":
    test_model(sys.argv[1])
