import json
import os

from tqdm import tqdm

# Define the folder path
folder_path = "data/01_raw/irishman"


def extract_id(abc_notation: str) -> str | None:
    """
    Extracts the id from the 'abc notation' field.
    Assumes the first line is in the format 'X:<id>'.
    """
    first_line = abc_notation.split("\n")[0]
    if first_line.startswith("X:"):
        return first_line[2:].strip()  # Get the string after "X:"
    return None


# File paths
train_file = os.path.join(folder_path, "train.json")
validation_file = os.path.join(folder_path, "validation.json")
leadsheet_ids_file = os.path.join(folder_path, "leadsheet_ids.json")

# Load the JSON data from files
with open(train_file, "r") as f:
    train_data = json.load(f)

with open(validation_file, "r") as f:
    validation_data = json.load(f)

with open(leadsheet_ids_file, "r") as f:
    leadsheet_ids = json.load(f)

# Create sets of ids for efficient lookup
train_ids = set(leadsheet_ids.get("train", []))
validation_ids = set(leadsheet_ids.get("validation", []))

# Filter the train samples using a loop with progress bar
leadsheet_train = []
for sample in tqdm(train_data, desc="Processing train samples"):
    if extract_id(sample.get("abc notation", "")) in train_ids:
        leadsheet_train.append(sample)

# Filter the validation samples using a loop with progress bar
leadsheet_validation = []
for sample in tqdm(validation_data, desc="Processing validation samples"):
    if extract_id(sample.get("abc notation", "")) in validation_ids:
        leadsheet_validation.append(sample)

# Output file paths
leadsheet_train_file = os.path.join(folder_path, "train_leadsheet.json")
leadsheet_validation_file = os.path.join(folder_path, "validation_leadsheet.json")

# Write the filtered samples to new JSON files
with open(leadsheet_train_file, "w") as f:
    json.dump(leadsheet_train, f, indent=4)

with open(leadsheet_validation_file, "w") as f:
    json.dump(leadsheet_validation, f, indent=4)
