import json
import os
import random


def split_json_file(input_filename: str, output_val: str, output_test: str) -> None:
    # Read the data from the input JSON file (assumed to be a list of samples)
    with open(input_filename, "r") as f:
        data = json.load(f)

    n = len(data)
    # Calculate the number of samples for the validation set.
    n_val = n // 2

    # Randomly select n_val indices for validation while preserving the original order.
    indices = list(range(n))
    val_indices = sorted(random.sample(indices, n_val))
    test_indices = sorted(set(indices) - set(val_indices))

    # Reconstruct each subset preserving the original order.
    data_val = [data[i] for i in val_indices]
    data_test = [data[i] for i in test_indices]

    # Write the subsets to their respective files.
    with open(output_val, "w") as f:
        json.dump(data_val, f, indent=4)
    with open(output_test, "w") as f:
        json.dump(data_test, f, indent=4)


# Folder path where all files are located
folder = "data/01_raw/irishman"

# Split validation.json into val.json and test.json
split_json_file(
    os.path.join(folder, "validation.json"), os.path.join(folder, "val.json"), os.path.join(folder, "test.json")
)

# Split leadsheet_validation.json into val_leadsheet.json and test_leadsheet.json
split_json_file(
    os.path.join(folder, "validation_leadsheet.json"),
    os.path.join(folder, "val_leadsheet.json"),
    os.path.join(folder, "test_leadsheet.json"),
)
