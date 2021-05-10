import json
import pickle
import sys

import nltk
import torch
import tqdm

import data
import models
from models import LanguageModel

MAX_LEN = 128
MAX_LEN_VADER = 40
BATCH_SIZE = 32
EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_params = torch.load(
    "models/training_checkpoint_oscar_vaderless.pt", map_location=device
)

model = models.LanguageModel(
    vocab_size=MAX_LEN,
    rnn_size=256,
    vader_size=MAX_LEN_VADER,
    use_vader=False,
    use_bert=False,
    use_cnn=False,
)
model.load_state_dict(model_params["model_state_dict"])
model = model.to(device)
model.eval()


def predict_stars(text):
    """
    text - a SINGLE texts
    """
    # This is where you call your model to get the number of stars output
    encodings = model.tokenizer.encode_plus(
        [text],
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        return_attention_mask=False,
        truncation=True,
        pad_to_max_length=False,
    )
    vectorized = encodings.get("input_ids", [])

    vadar_sentiments = nltk.tokenize.sent_tokenize(text)

    # Place the data as a batch, even if there is only 1
    vectorized = [vectorized]
    vadar_sentiments = [vadar_sentiments]

    return model.predict(vectorized, vadar_sentiments)


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
                            "predicted_stars": predict_stars(review["text"]),
                        }
                    )
                    + "\n"
                )
    print("Output prediction file written")
else:
    print("No validation file given")
