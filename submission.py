import json
from models import LanguageModel
import sys
import torch
import models
import nltk


def eval(text):
    # This is where you call your model to get the number of stars output
    model_params = torch.load("models/submission.pt", map_location=torch.device('cpu'))
    model = models.LanguageModel(
        vocab_size=128,     
        rnn_size=256,
        vader_size=40,
        use_vader=True,
        use_bert=False,
        use_cnn=False,
    )
    model.eval()
    return model(text)


if len(sys.argv) > 1:
    validation_file = sys.argv[1]
    with open("output.jsonl", "w") as fw:
        with open(validation_file, "r") as fr:
            for line in fr:
                review = json.loads(line)
                fw.write(
                    json.dumps(
                        {
                            "review_id": review["review_id"],
                            "predicted_stars": eval(review["text"]),
                        }
                    )
                    + "\n"
                )
    print("Output prediction file written")
else:
    print("No validation file given")
