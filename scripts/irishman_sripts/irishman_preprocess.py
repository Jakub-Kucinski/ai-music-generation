import json
import os

from tqdm import tqdm

# Define input and output directories
input_dir = "data/01_raw/irishman"
output_dir = "data/02_preprocessed/irishman"
os.makedirs(output_dir, exist_ok=True)

# List of files to process
files = [
    "train.json",
    "validation.json",
    "val.json",
    "test.json",
    "train_leadsheet.json",
    "validation_leadsheet.json",
    "val_leadsheet.json",
    "test_leadsheet.json",
]

for filename in tqdm(files):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    with open(input_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    for item in tqdm(data, leave=False):
        abc = item.get("abc notation", "")
        # Check if the string starts with "X:" and extract id
        if abc.startswith("X:"):
            newline_index = abc.find("\n")
            if newline_index != -1:
                # Extract id from the header (remove "X:" prefix)
                id_val = abc[2:newline_index].strip()
                item["id"] = id_val
                # Remove the "X:id\n" header from the abc notation field
                item["abc notation"] = abc[newline_index + 1 :]

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4)
