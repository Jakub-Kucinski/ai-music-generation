#!/usr/bin/env python3
import json
import multiprocessing
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd
from midi2audio import FluidSynth
from sox.transform import Transformer as SoxTransformer
from tqdm import tqdm

# Converter settings
midi_to_wav_converter: Literal["Timidity", "FluidSynth"] = "FluidSynth"
sound_fonts_dir = "sound_fonts"
sound_font: str | None = "Essential Keys-sforzando-v9.6.sf2"
sample_rate = 16_000

# Define paths
midi_input_folder = "data/04_generated/irishman_midi/unconditioned_samples/midi"
base_output_dir = "data/04_generated/irishman_midi/unconditioned_samples"

# Create subdirectory for WAV outputs
wav_output_dir = os.path.join(
    base_output_dir,
    "wav",
    midi_to_wav_converter,
    f"{sound_font if sound_font else 'default'}",
)
os.makedirs(wav_output_dir, exist_ok=True)

# Prepare folder for the JSONL files (WAV paths and aesthetics)
audiobox_dir = os.path.join(
    base_output_dir,
    "audiobox_aesthetics",
    midi_to_wav_converter,
    f"{sound_font if sound_font else 'default'}",
)
os.makedirs(audiobox_dir, exist_ok=True)
input_jsonl_filename = os.path.join(audiobox_dir, "wav_paths.jsonl")
output_jsonl_filename = os.path.join(audiobox_dir, "aesthetics.jsonl")
output_aggregated_aesthetics = os.path.join(audiobox_dir, "aesthetics_aggregated.jsonl")


def process_midi_file(midi_filename: str) -> str:
    midi_file_path = os.path.join(midi_input_folder, midi_filename)
    # Use the file name (without extension) as the index
    idx = os.path.splitext(midi_filename)[0]

    # Define output WAV file path
    wav_file_path = os.path.join(wav_output_dir, f"file_{idx}.wav")

    # # Skip conversion if the WAV file already exists
    # if os.path.exists(wav_file_path):
    #     return os.path.abspath(wav_file_path)

    # Convert MIDI to WAV
    if midi_to_wav_converter == "Timidity":
        subprocess.run(
            ["timidity", midi_file_path, "-Ow", "-o", wav_file_path, "-s", str(sample_rate)],
            check=False,
        )
    elif midi_to_wav_converter == "FluidSynth":
        if sound_font:
            fs = FluidSynth(sound_font=os.path.join(sound_fonts_dir, sound_font), sample_rate=sample_rate)
        else:
            fs = FluidSynth(sample_rate=sample_rate)
        fs.midi_to_audio(midi_file_path, wav_file_path)

    # Remove silence at the end of wav file produced by SoundFont configuration
    transformer = SoxTransformer().silence(
        location=-1,
        silence_threshold=0.1,
        min_silence_duration=0.1,
        buffer_around_silence=False,
    )
    # 2) Create a temp file next to the original
    with tempfile.NamedTemporaryFile(dir=wav_output_dir, suffix=Path(wav_file_path).suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # 3) Write the processed audio to the temp file
        transformer.build(wav_file_path, str(tmp_path))

        # 4) Atomically replace the original
        os.replace(tmp_path, wav_file_path)  # overwrites if target exists
    finally:
        # 5) If anything failed, make sure the temp file is removed
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return os.path.abspath(wav_file_path)


if __name__ == "__main__":
    # Get a list of all MIDI files in the input folder
    midi_files = sorted([f for f in os.listdir(midi_input_folder) if f.endswith(".mid")])
    print(midi_files)

    # Process each MIDI file using multiprocessing
    with multiprocessing.Pool() as pool:
        wav_paths = list(tqdm(pool.imap(process_midi_file, midi_files), total=len(midi_files)))

    # Write the collected WAV file paths to a JSONL file
    with open(input_jsonl_filename, "w") as out_file:
        for path in wav_paths:
            json_line = json.dumps({"path": path})
            out_file.write(json_line + "\n")

    print(f"\nWAV file paths saved to {input_jsonl_filename}")

    # Run the aesthetics calculation and write output to a JSONL file
    with open(output_jsonl_filename, "w") as outfile:
        subprocess.run(["audio-aes", input_jsonl_filename, "--batch-size", "10"], stdout=outfile)

    # Load the aesthetics JSONL file into a DataFrame
    df = pd.read_json(output_jsonl_filename, lines=True)

    # Compute mean and standard deviation for selected aesthetic columns
    stats = df[["CE", "CU", "PC", "PQ"]].agg(["mean", "std"])

    # Save the aggregated aesthetics statistics to a JSON file
    stats.to_json(output_aggregated_aesthetics, orient="index", indent=4)

    print(stats)
