#!/usr/bin/env python3
import json
import os
import re
import subprocess

from tqdm import tqdm

# Define paths
input_json = "data/01_raw/irishman/validation_leadsheet.json"
base_output_dir = "data/02_preprocessed/irishman"

# Create subdirectories for each file type inside a validation_leadsheet folder
abc_output_dir = os.path.join(base_output_dir, "abc", "validation_leadsheet")
midi_output_dir = os.path.join(base_output_dir, "midi", "validation_leadsheet")
wav_output_dir = os.path.join(base_output_dir, "wav", "validation_leadsheet")

os.makedirs(abc_output_dir, exist_ok=True)
os.makedirs(midi_output_dir, exist_ok=True)
os.makedirs(wav_output_dir, exist_ok=True)

# Prepare folder for the JSONL file with WAV paths
audiobox_dir = os.path.join(base_output_dir, "audiobox_aesthetics")
os.makedirs(audiobox_dir, exist_ok=True)
input_jsonl_filename = os.path.join(audiobox_dir, "wav_paths.jsonl")
output_jsonl_filename = os.path.join(audiobox_dir, "aesthetics.jsonl")

# Load JSON data
with open(input_json, "r") as f:
    leadsheets = json.load(f)

# List to collect the WAV file paths
wav_paths = []

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
    wav_file_path = os.path.join(wav_output_dir, f"file_{idx}.wav")

    # Convert the abc file to a MIDI file using abc2midi
    subprocess.run(["abc2midi", abc_file_path, "-o", midi_file_path], check=False)

    # Convert the MIDI file to a WAV file using timidity
    subprocess.run(["timidity", midi_file_path, "-Ow", "-o", wav_file_path], check=False)

    # Append the absolute WAV file path to the list
    wav_paths.append(os.path.abspath(wav_file_path))


# Write the collected WAV file paths to the JSONL file
with open(input_jsonl_filename, "w") as out_file:
    for path in wav_paths:
        json_line = json.dumps({"path": path})
        out_file.write(json_line + "\n")

print(f"\nWAV file paths saved to {input_jsonl_filename}")

with open(output_jsonl_filename, "w") as outfile:
    subprocess.run(["audio-aes", input_jsonl_filename, "--batch-size", "10"], stdout=outfile)
