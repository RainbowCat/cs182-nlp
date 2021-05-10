import json
import pickle
import sys

import nltk
import torch
import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import data
import models
from models import LanguageModel

MAX_LEN = 128
MAX_LEN_VADER = 40
BATCH_SIZE = 32
EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_params = torch.load(
    "models/training_checkpoint_oscar_vader.pt", map_location=device
)

model = models.LanguageModel(
    vocab_size=MAX_LEN,
    rnn_size=256,
    vader_size=MAX_LEN_VADER,
    use_vader=True,
    use_bert=False,
    use_cnn=True,
)
model.load_state_dict(model_params["model_state_dict"])
model = model.to(device)
model.eval()

analyzer = SentimentIntensityAnalyzer()


def predict_stars(text):
    """
    text - a SINGLE texts
    """
    # This is where you call your model to get the number of stars output
    encodings = model.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        return_attention_mask=False,
        truncation=True,
        pad_to_max_length=False,
    )
    text_encoding = encodings.get("input_ids", [])
    vectorized, _ = data.pad_sequence(text_encoding, 0, MAX_LEN)

    sentence_list = nltk.tokenize.sent_tokenize(
        text
    )  # Text is one at a time anyway here
    review_sentiment_sentence = []
    for sentence in sentence_list:
        vs = analyzer.polarity_scores(sentence)
        review_sentiment_sentence.append(vs["compound"])
    vadar_sentiments, _ = data.pad_sequence(review_sentiment_sentence, 0, MAX_LEN_VADER)

    # Place the data as a batch, even if there is only 1
    vectorized = data.batch_to_torch_long([vectorized])
    vadar_sentiments = data.batch_to_torch_float([vadar_sentiments])

    p = model.predict(vectorized, vadar_sentiments)
    print(p, p[0], p[0].item())
    return float(p[0].item())


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
