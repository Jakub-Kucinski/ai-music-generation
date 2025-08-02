import os
import pickle
import random

import numpy as np


def load_texts_from_directory(directory: str) -> list[str]:
    """
    Loads the content of all .txt files in the given directory into a list.

    Args:
        directory (str): The directory from which to load the .txt files.

    Returns:
        list of str: A list where each element is the content of a text file.
    """
    texts: list[str] = []
    # Loop over each file in the specified directory
    for filename in os.listdir(directory):
        # Process only files that end with '.txt'
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(directory, filename)
            # Open and read the file, then append its contents to the list.
            with open(filepath, "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts


# Define the directories for validation and training texts
validation_dir = "data/03_converted/music21_bach/validation/midi_texts"
train_dir = "data/03_converted/music21_bach/train/midi_texts_augmented"
output_dir = "data/03_converted/music21_bach"

# Load the text files into lists
valid_data = load_texts_from_directory(validation_dir)
train_data = load_texts_from_directory(train_dir)
random.shuffle(train_data)
random.shuffle(valid_data)

# Output a summary of the results
print(f"Loaded {len(valid_data)} texts from the validation directory.")
print(f"Loaded {len(train_data)} texts from the train directory.")


for i in range(len(valid_data)):
    valid_data[i] = "$ " + valid_data[i]

for i in range(len(train_data)):
    train_data[i] = "$ " + train_data[i]

train_text = " ".join(train_data)
valid_text = " ".join(valid_data)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Write the texts to files
with open(os.path.join(output_dir, "train_text_augmented.txt"), "w") as f:
    f.write(train_text)

with open(os.path.join(output_dir, "valid_text_not_augmented.txt"), "w") as f:
    f.write(valid_text)

# 4. Split both texts by whitespace into lists
train_tokens = train_text.split()
valid_tokens = valid_text.split()

# 5. Create a single list of all unique tokens that appear in both lists (the union)
unique_tokens = list(set(train_tokens) | set(valid_tokens))
vocab_size = len(unique_tokens)

# 6. Create mappings using enumerate: index-to-token and token-to-index
itos = {i: token for i, token in enumerate(unique_tokens)}
stoi = {token: i for i, token in itos.items()}

# Example: Print the size of the vocabulary and a couple of mappings
print("Vocabulary size:", vocab_size)
print("Sample itos mapping:", dict(list(itos.items())[:5]))
print("Sample stoi mapping:", dict(list(stoi.items())[:5]))


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s.split()]  # encoder: take a string, output a list of integers


def decode(list_of_tokens_ids: list[int]) -> str:
    return " ".join([itos[i] for i in list_of_tokens_ids])  # decoder: take a list of integers, output a string


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

# Loaded 32 texts from the validation directory.
# Loaded 3504 texts from the train directory.
# Vocabulary size: 121
# Sample itos mapping: {0: 'p42', 1: 'p57', 2: 'time_signature_4/4', 3: 'p48', 4: 'p30'}
# Sample stoi mapping: {'p42': 0, 'p57': 1, 'time_signature_4/4': 2, 'p48': 3, 'p30': 4}
# train has 3,340,464 tokens
# val has 30,031 tokens
