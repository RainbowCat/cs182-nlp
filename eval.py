import json
import sys

import nltk
import torch
from nltk import text

import data
from models import LanguageModel
from utils import *


def eval(text):
    # This is where you call your model to get the number of stars output
    model_params = torch.load("models/submission.pt", map_location=torch.device("cpu"))
    model = LanguageModel(
        vocab_size=MAX_LEN,
        rnn_size=256,
        vader_size=MAX_LEN_VADER,
        use_vader=True,
        use_bert=False,
        use_cnn=False,
    )
    model.load_state_dict(model_params["state_dict"])
    model.eval()
    vectorized = data.tokenize(text)
    vadar_sentiments = nltk.tokenize.sent_tokenize([text])
    return model.predict(vectorized, vadar_sentiments)


if len(sys.argv) > 1:
    text = sys.argv[1]
    print(eval(text))
#     with open("output.jsonl", "w") as fw:
#         with open(validation_file, "r") as fr:
#             for line in fr:
#                 review = json.loads(line)
#                 fw.write(
#                     json.dumps(
#                         {
#                             "review_id": review["review_id"],
#                             "predicted_stars": eval(review["text"]),
#                         }
#                     )
#                     + "\n"
#                 )
#     print("Output prediction file written")
# else:
#     print("No validation file given")
