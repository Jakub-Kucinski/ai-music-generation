#!/usr/bin/env python3
import json
import multiprocessing
import os
import re
import subprocess
import tempfile
from pathlib import Path
from statistics import NormalDist
from typing import Literal

import music21
import pandas as pd
from midi2audio import FluidSynth
from sox.transform import Transformer as SoxTransformer
from tqdm import tqdm

# Converter settings
abc_to_midi_converter: Literal["music21", "abc2midi"] = "abc2midi"
midi_to_wav_converter: Literal["Timidity", "FluidSynth"] = "FluidSynth"
sound_fonts_dir = "sound_fonts"
sound_font: str | None = "Essential Keys-sforzando-v9.6.sf2"
sample_rate = 16_000

# Define paths
abc_input_folder = "data/04_generated/irishman_1k_context/unconditioned/abc"
base_output_dir = "data/04_generated/irishman_1k_context/unconditioned"
os.makedirs(base_output_dir, exist_ok=True)

# Create subdirectories for MIDI and WAV outputs
midi_output_dir = os.path.join(base_output_dir, "midi", abc_to_midi_converter)
wav_output_dir = os.path.join(
    base_output_dir,
    "wav",
    abc_to_midi_converter,
    midi_to_wav_converter,
    f"{sound_font if sound_font else 'default'}",
)
os.makedirs(midi_output_dir, exist_ok=True)
os.makedirs(wav_output_dir, exist_ok=True)

# Prepare folder for the JSONL files (WAV paths and aesthetics)
audiobox_dir = os.path.join(
    base_output_dir,
    "audiobox_aesthetics",
    abc_to_midi_converter,
    midi_to_wav_converter,
    f"{sound_font if sound_font else 'default'}",
)
os.makedirs(audiobox_dir, exist_ok=True)
input_jsonl_filename = os.path.join(audiobox_dir, "wav_paths.jsonl")
output_jsonl_filename = os.path.join(audiobox_dir, "aesthetics.jsonl")
output_aggregated_aesthetics = os.path.join(audiobox_dir, "aesthetics_aggregated.jsonl")

# List to collect the absolute WAV file paths
wav_paths = []


def process_abc_file(abc_filename: str) -> str:
    abc_file_path = os.path.join(abc_input_folder, abc_filename)
    with open(abc_file_path, "r") as file:
        abc_content = file.read()

    # Try to extract the index from a line starting with "X:"
    match = re.search(r"^X:\s*(\d+)", abc_content, flags=re.MULTILINE)
    if match:
        idx = match.group(1)
    else:
        # Fallback: use the file name (without extension) as the index
        idx = os.path.splitext(abc_filename)[0].split("_")[-1]

    # Define output file paths
    midi_file_path = os.path.join(midi_output_dir, f"file_{idx}.mid")
    wav_file_path = os.path.join(wav_output_dir, f"file_{idx}.wav")

    # Convert ABC to MIDI
    if abc_to_midi_converter == "abc2midi":
        subprocess.run(["abc2midi", abc_file_path, "-o", midi_file_path], check=False)
    elif abc_to_midi_converter == "music21":
        score = music21.converter.parse(abc_file_path)
        score.write("midi", fp=midi_file_path)

    # Convert MIDI to WAV
    if midi_to_wav_converter == "Timidity":
        subprocess.run(["timidity", midi_file_path, "-Ow", "-o", wav_file_path, "-s", str(sample_rate)], check=False)
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
    # Get a list of all ABC files in the input folder
    abc_files = sorted([f for f in os.listdir(abc_input_folder) if f.endswith(".abc")])

    # Process each ABC file using multiprocessing
    with multiprocessing.Pool() as pool:
        wav_paths = list(tqdm(pool.imap(process_abc_file, abc_files), total=len(abc_files)))

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

    mean = df.mean()  # column‑wise means
    se = df.sem(ddof=1)  # column‑wise standard errors = s / √n

    Z_95 = NormalDist().inv_cdf(0.975)
    moe = Z_95 * se  # calculate margin of error for 95% confidence interval
    ci_lower = mean - moe
    ci_upper = mean + moe

    # 3) build output dict including margin-of-error and ci95
    out = {
        "mean": mean.to_dict(),
        "se": se.to_dict(),
        "moe": moe.to_dict(),
        "ci95_lower": ci_lower.to_dict(),
        "ci95_upper": ci_upper.to_dict(),
    }

    # 4) write JSON summary
    with open(output_aggregated_aesthetics, "w") as f:
        json.dump(out, f, indent=4)

    # 5) optional console print
    print("\nMean ± MoE (95% CI half-width)")
    print("-" * 40)
    for col in mean.index:
        print(f"{col:>3}: {mean[col]:.6f} ± {moe[col]:.6f}  " f"[{ci_lower[col]:.6f}, {ci_upper[col]:.6f}]")
