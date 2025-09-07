import os
import pickle
import random

import numpy as np
from tqdm import tqdm


def load_texts_from_directory(
    directory: str, *, followlinks: bool = False, shuffle_within_subdirs: bool = True, seed: int | None = None
) -> list[str]:
    """
    Loads the content of all .txt files in the given directory and subdirectories.

    Args:
        directory (str): Root directory to search.
        followlinks (bool): Whether to follow symbolic links. Defaults to False.
        shuffle_within_subdirs (bool): Whether to shuffle files within each subdirectory. Defaults to True.
        seed (int): Random seed for reproducibility. Defaults to None.

    Returns:
        list[str]: Contents of each .txt file found.
    """
    texts: list[str] = []

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Group files by directory
    files_by_directory: dict[str, list[str]] = {}

    for root, _, files in os.walk(directory, followlinks=followlinks):
        txt_files_in_dir = []
        for name in files:
            if name.lower().endswith(".txt"):
                txt_files_in_dir.append(os.path.join(root, name))

        if txt_files_in_dir:  # Only add if there are txt files
            files_by_directory[root] = txt_files_in_dir

    # Process each directory's files
    all_txt_files = []
    for dir_path, file_paths in sorted(files_by_directory.items()):  # Sort by directory for consistency
        if shuffle_within_subdirs:
            random.shuffle(file_paths)
        all_txt_files.extend(file_paths)

    # Process files with progress bar
    for path in tqdm(all_txt_files, desc="Loading text files", unit="file"):
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    return texts


# Define the directories for validation and training texts
validation_dir = "data/03_converted/synthetic_4_parts_from_irishman/validation/midi_texts"
train_dir = "data/03_converted/synthetic_4_parts_from_irishman/train/midi_texts"
output_dir = "data/03_converted/synthetic_4_parts_from_irishman"

# Load the text files into lists with shuffling within subdirectories
# Use a seed for reproducibility
valid_data = load_texts_from_directory(validation_dir, shuffle_within_subdirs=True, seed=42)
train_data = load_texts_from_directory(train_dir, shuffle_within_subdirs=True, seed=42)

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
with open(os.path.join(output_dir, "train_text.txt"), "w") as f:
    f.write(train_text)

with open(os.path.join(output_dir, "valid_text.txt"), "w") as f:
    f.write(valid_text)

with open("data/03_converted/music21_bach/train_text.txt", "r") as f:
    bach_train = f.read()
with open("data/03_converted/music21_bach/valid_text.txt", "r") as f:
    bach_valid = f.read()
bach_train_tokens = bach_train.split()
bach_valid_tokens = bach_valid.split()

# 4. Split both texts by whitespace into lists
train_tokens = train_text.split()
valid_tokens = valid_text.split()

# 5. Create a single list of all unique tokens that appear in both lists (the union)
# unique_tokens = list(set(train_tokens) | set(valid_tokens))
unique_tokens = list(set(train_tokens) | set(valid_tokens) | set(bach_train_tokens) | set(bach_valid_tokens))
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

# Loaded 34 texts from the validation directory.
# Loaded 305 texts from the train directory.
# Vocabulary size: 121
# Sample itos mapping: {0: 'p84', 1: '/', 2: 'time_signature_3/4', 3: 'd72', 4: 'o21'}
# Sample stoi mapping: {'p84': 0, '/': 1, 'time_signature_3/4': 2, 'd72': 3, 'o21': 4}
# train has 310,183 tokens
# val has 30,509 tokens
