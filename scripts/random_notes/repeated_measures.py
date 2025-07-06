import json
import os
import random
import re

from tqdm import tqdm

NUM_FILES = 1000  # Number of ABC files to generate
OUTPUT_DIR = "data/04_generated/irishman/repeated_measures/abc"

JSON_FILE_PATH = "data/02_preprocessed/irishman/validation_leadsheet.json"
NUM_MEASURES = 32


def load_abc_entries(json_file: str) -> tuple[list[str], list[str]]:
    with open(json_file, "r") as f:
        data = json.load(f)
    # This regex will later be used to split the measures part on common bar symbols.
    measure_regex = re.compile(r"\s*(?:\[\||\|:|::|:\||\|\|+|\||\|\])\s*")
    descriptions = []
    measures = []
    # Loop through each entry in the JSON file
    for entry in data:
        abc_text: str = entry.get("abc notation", "")
        # Use the key signature line as the end of the description part
        match = re.search(r"\nK:.*\n", abc_text)
        if match:
            # description_part includes up to and including the key line
            description_part = abc_text[: match.end()]
            measures_part = abc_text[match.end() :]
        else:
            # If no key signature found, skip this entry
            continue
        description_part = description_part.strip()
        descriptions.append(description_part)
        # Split the remaining text into measures using the regex
        parts = measure_regex.split(measures_part)
        for part in parts:
            measure = part.strip()
            if measure:
                measures.append(measure)
    return descriptions, measures


def create_new_abc(descriptions: list[str], measures: list[str], idx: int, num_measures: int) -> str:
    # Randomly select one description from the list
    selected_description = random.choice(descriptions)
    match = re.search(r"^X:\s*(\d+)", selected_description, flags=re.MULTILINE)
    if match:
        selected_description = re.sub(r"^X:\s*(\d+)", f"X:{idx}", selected_description, flags=re.MULTILINE)
    else:
        selected_description = f"X:{idx}\n" + selected_description
    # Randomly select ONE measure and repeat it for all measures
    selected_measure = random.choice(measures)
    selected_measures = [selected_measure] * num_measures
    # Join the selected measures with a bar symbol
    measures_str = " | ".join(selected_measures)
    # Combine the selected description and measures
    new_abc = selected_description.strip() + "\n" + measures_str.strip()
    return new_abc


if __name__ == "__main__":
    descriptions, measures = load_abc_entries(JSON_FILE_PATH)
    if not measures:
        print("No measures were found in the provided JSON file.")
    else:
        # Create the output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Generate and save multiple ABC files
        for i in tqdm(range(NUM_FILES)):
            new_abc = create_new_abc(descriptions, measures, i + 1, NUM_MEASURES)
            file_path = os.path.join(OUTPUT_DIR, f"file_{i+1}.abc")
            with open(file_path, "w") as f:
                f.write(new_abc)
