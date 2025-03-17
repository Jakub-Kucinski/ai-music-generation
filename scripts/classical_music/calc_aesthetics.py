#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path

import pandas as pd


def main(root_folder: str, audiobox_dir: str) -> None:
    os.makedirs(audiobox_dir, exist_ok=True)
    input_jsonl_filename = os.path.join(audiobox_dir, "wav_paths.jsonl")
    output_jsonl_filename = os.path.join(audiobox_dir, "aesthetics.jsonl")
    output_aggregated_aesthetics = os.path.join(audiobox_dir, "aesthetics_aggregated.jsonl")
    root = Path(root_folder)
    with open(input_jsonl_filename, "w") as outfile:
        # Recursively search for all .wav files
        for wav_file in root.rglob("*.wav"):
            # Write each file path in JSONL format
            json_line = json.dumps({"path": str(wav_file.resolve())})
            outfile.write(json_line + "\n")

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


if __name__ == "__main__":
    input_folder = "data/03_converted/Classical Music MIDI"
    audiobox_dir = audiobox_dir = os.path.join(
        input_folder,
        "audiobox_aesthetics",
    )
    main(input_folder, audiobox_dir)
