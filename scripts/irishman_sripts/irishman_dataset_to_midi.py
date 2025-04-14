#!/usr/bin/env python3
import json
import os
import re
import subprocess
from typing import Literal

import music21
from tqdm import tqdm

# Converter settings
abc_to_midi_converter: Literal["music21", "abc2midi"] = "abc2midi"

# Define paths
input_json = "data/01_raw/irishman/train_leadsheet.json"
base_output_dir = "data/03_converted/irishman/train_leadsheet"
os.makedirs(base_output_dir, exist_ok=True)

# Create subdirectories for each file type inside a validation_leadsheet folder
abc_output_dir = os.path.join(base_output_dir, "abc")
midi_output_dir = os.path.join(base_output_dir, "midi", abc_to_midi_converter)

os.makedirs(abc_output_dir, exist_ok=True)
os.makedirs(midi_output_dir, exist_ok=True)


# Load JSON data
with open(input_json, "r") as f:
    leadsheets = json.load(f)

# Process each leadsheet entry
for sheet in tqdm(leadsheets):
    abc_notation = sheet.get("abc notation")
    if not abc_notation:
        continue  # Skip if no abc notation is provided

    # Extract the index from the first line starting with "X:"
    match = re.search(r"^X:\s*(\d+)", abc_notation, flags=re.MULTILINE)
    if match:
        idx = match.group(1)
    else:
        print("Skipping entry without valid 'X:' field.")
        continue

    # Save the abc notation to a file in the abc output folder
    abc_file_path = os.path.join(abc_output_dir, f"file_{idx}.abc")
    with open(abc_file_path, "w") as abc_file:
        abc_file.write(abc_notation)

    # Define output file paths using the extracted index
    midi_file_path = os.path.join(midi_output_dir, f"file_{idx}.mid")

    # Convert the abc file to a MIDI file using abc2midi
    if abc_to_midi_converter == "abc2midi":
        subprocess.run(["abc2midi", abc_file_path, "-o", midi_file_path], check=False)
    elif abc_to_midi_converter == "music21":
        score = music21.converter.parse(abc_file_path)
        score.write("midi", fp=midi_file_path)
