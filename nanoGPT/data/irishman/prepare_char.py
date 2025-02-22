import json
import os
import pickle

import numpy as np

# Define the data directory
data_dir = "data/02_preprocessed/irishman"

# 1. Load the data from the specified directory
with open(os.path.join(data_dir, "train.json"), "r") as f:
    train_data = json.load(f)

with open(os.path.join(data_dir, "validation.json"), "r") as f:
    valid_data = json.load(f)

# 2. Prepend "<start> " and append " <end>" to each "abc notation"
for entry in train_data:
    # Strip any surrounding whitespace to be safe
    entry["abc notation"] = "<start> " + entry["abc notation"].strip() + " <end>"

for entry in valid_data:
    entry["abc notation"] = "<start> " + entry["abc notation"].strip() + " <end>"

# 3. Join all "abc notation" strings from train and from validation using a space
train_text = " ".join(entry["abc notation"] for entry in train_data)
valid_text = " ".join(entry["abc notation"] for entry in valid_data)

# Save train_text and valid_text to the data/02_preprocessed/irishman folder

# Ensure the output directory exists
output_dir = "data/02_preprocessed/irishman"
os.makedirs(output_dir, exist_ok=True)

# Write the texts to files
with open(os.path.join(output_dir, "train_text.txt"), "w") as f:
    f.write(train_text)

with open(os.path.join(output_dir, "valid_text.txt"), "w") as f:
    f.write(valid_text)

# 4. Create a single list of all unique tokens that appear in both lists (the union)
unique_tokens = list(set(train_text + valid_text))
vocab_size = len(unique_tokens)

# 5. Create mappings using enumerate: index-to-token and token-to-index
itos = {i: token for i, token in enumerate(unique_tokens)}
stoi = {token: i for i, token in itos.items()}

# Example: Print the size of the vocabulary and a couple of mappings
print("Vocabulary size:", vocab_size)
print("Sample itos mapping:", dict(list(itos.items())[:5]))
print("Sample stoi mapping:", dict(list(stoi.items())[:5]))


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(list_of_tokens_ids: list[int]) -> str:
    return "".join([itos[i] for i in list_of_tokens_ids])  # decoder: take a list of integers, output a string


# encode both to integers
train_ids = encode(train_text)
val_ids = encode(valid_text)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids_np = np.array(train_ids, dtype=np.uint16)
val_ids_np = np.array(val_ids, dtype=np.uint16)
train_ids_np.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids_np.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as file:
    pickle.dump(meta, file)
