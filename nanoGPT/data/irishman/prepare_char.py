import json
import os
import pickle

import numpy as np

# Define the data directory
data_dir = "data/02_preprocessed/irishman"
# train_file = "train.json"
# validation_file = "validation.json"
train_file = "train_leadsheet.json"
validation_file = "validation_leadsheet.json"

# 1. Load the data from the specified directory
with open(os.path.join(data_dir, train_file), "r") as f:
    train_data = json.load(f)

with open(os.path.join(data_dir, validation_file), "r") as f:
    valid_data = json.load(f)


for entry in train_data:
    # Strip any surrounding whitespace to be safe
    entry["abc notation"] = entry["abc notation"].strip() + " $"

for entry in valid_data:
    entry["abc notation"] = entry["abc notation"].strip() + " $"


train_lengths = [len(entry["abc notation"]) for entry in train_data]
valid_lengths = [len(entry["abc notation"]) for entry in valid_data]

# Define which percentiles to compute
percentiles = [0, 10, 25, 50, 75, 90, 100]

# Compute percentiles using numpy
train_percentiles = np.percentile(train_lengths, percentiles)
valid_percentiles = np.percentile(valid_lengths, percentiles)

print("Train 'abc notation' length percentiles:")
for p, val in zip(percentiles, train_percentiles):
    print(f"  {p}th percentile: {val}")

print("\nValidation 'abc notation' length percentiles:")
for p, val in zip(percentiles, valid_percentiles):
    print(f"  {p}th percentile: {val}")


# 3. Join all "abc notation" strings from train and from validation using a space
train_text = " ".join(entry["abc notation"] for entry in train_data)
valid_text = " ".join(entry["abc notation"] for entry in valid_data)

# Save train_text and valid_text to the data/02_preprocessed/irishman folder

# Ensure the output directory exists
output_dir = "data/02_preprocessed/irishman"
os.makedirs(output_dir, exist_ok=True)

# # Write the texts to files
# with open(os.path.join(output_dir, "train_text.txt"), "w") as f:
#     f.write(train_text)

# with open(os.path.join(output_dir, "valid_text.txt"), "w") as f:
#     f.write(valid_text)

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


# train_file = "train.json"
# validation_file = "validation.json"
# Train 'abc notation' length percentiles:
#   0th percentile: 17.0
#   10th percentile: 160.0
#   25th percentile: 201.0
#   50th percentile: 250.0
#   75th percentile: 328.0
#   90th percentile: 443.0
#   100th percentile: 2961.0

# Validation 'abc notation' length percentiles:
#   0th percentile: 61.0
#   10th percentile: 161.0
#   25th percentile: 201.0
#   50th percentile: 250.0
#   75th percentile: 330.0
#   90th percentile: 435.0
#   100th percentile: 1079.0
# Vocabulary size: 95
# Sample itos mapping: {0: 'K', 1: 'f', 2: ':', 3: '^', 4: '`'}
# Sample stoi mapping: {'K': 0, 'f': 1, ':': 2, '^': 3, '`': 4}
# train has 60,944,859 tokens
# val has 611,888 tokens


# train_file = "train_leadsheet.json"
# validation_file = "validation_leadsheet.json"
# Train 'abc notation' length percentiles:
#   0th percentile: 29.0
#   10th percentile: 233.0
#   25th percentile: 277.0
#   50th percentile: 334.0
#   75th percentile: 445.0
#   90th percentile: 591.0
#   100th percentile: 2088.0

# Validation 'abc notation' length percentiles:
#   0th percentile: 126.0
#   10th percentile: 238.9
#   25th percentile: 280.5
#   50th percentile: 341.0
#   75th percentile: 433.75
#   90th percentile: 576.8
#   100th percentile: 940.0
# Vocabulary size: 95
# Sample itos mapping: {0: 'A', 1: 'h', 2: 'R', 3: '[', 4: 'k'}
# Sample stoi mapping: {'A': 0, 'h': 1, 'R': 2, '[': 3, 'k': 4}
# train has 12,935,627 tokens
# val has 117,776 tokens
