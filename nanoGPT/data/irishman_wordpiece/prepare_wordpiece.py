import json
import os
from typing import cast

import numpy as np
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

VOCAB_SIZE = 1024
old_tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

# Define the data directory
data_dir = "data/02_preprocessed/irishman"
train_file = "train.json"
validation_file = "validation.json"

# 1. Load the data from the specified directory
with open(os.path.join(data_dir, train_file), "r") as f:
    train_data = json.load(f)

with open(os.path.join(data_dir, validation_file), "r") as f:
    valid_data = json.load(f)


for entry in train_data:
    # Strip any surrounding whitespace to be safe
    entry["abc notation"] = entry["abc notation"].strip()

for entry in valid_data:
    entry["abc notation"] = entry["abc notation"].strip()

# Join all "abc notation" strings from train and from validation using a space
train_texts = [entry["abc notation"] for entry in train_data]
valid_texts = " ".join(entry["abc notation"] for entry in valid_data)

# Train new tokenizer
texts = [entry["abc notation"] for entry in train_data + valid_data]
tokenizer = cast(BertTokenizerFast, old_tokenizer.train_new_from_iterator([texts], VOCAB_SIZE))


def encode_text(s: str) -> list[int]:
    encoding = tokenizer.encode(s)
    encoding.append(tokenizer.vocab["[SEP]"])
    return encoding  # type: ignore


# encode both to integers
train_ids = []
for text in train_texts:
    train_ids.extend(encode_text(text))

val_ids = []
for text in valid_texts:
    val_ids.extend(encode_text(text))

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# # export to bin files
train_ids_np = np.array(train_ids, dtype=np.uint16)
val_ids_np = np.array(val_ids, dtype=np.uint16)
train_ids_np.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids_np.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))


# save tokenizer
tokenizer.save_vocabulary(os.path.join(os.path.dirname(__file__), "vocab.json"))
