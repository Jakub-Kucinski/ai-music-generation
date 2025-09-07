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

from ai_music_generation.core.metrics.calculate import (
    SimilarityResult,
    calculate_conditioned_similarity_of_music_vectors,
    calculate_inner_similarity_of_music_vectors,
    calculate_reference_similarity_of_music_vectors,
)
from ai_music_generation.core.metrics.similarities import (
    cyclic_pitch_similarity,
    rhythmic_similarity,
)
from ai_music_generation.core.metrics.vectorization import MidiVectorizer

# Converter settings
abc_to_midi_converter: Literal["music21", "abc2midi"] = "abc2midi"
midi_to_wav_converter: Literal["Timidity", "FluidSynth"] = "FluidSynth"
sound_fonts_dir = "sound_fonts"
sound_font: str | None = "Essential Keys-sforzando-v9.6.sf2"
sample_rate = 16_000

reference_midi_files_dir: str | None = None
reference_midi_files_dir = "data/03_converted/irishman/validation_leadsheet/midi/abc2midi"
n_conditioned_measures: int = 0
n_conditioned_measures = 4

# Define paths
abc_input_folder = "data/04_generated/tunesformer/conditioned_4_bars/abc"
base_output_dir = "data/04_generated/tunesformer/conditioned_4_bars"
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

# Create subdirectory for metrics
metrics_dir = os.path.join(
    base_output_dir,
    "metrics",
    abc_to_midi_converter,
)
os.makedirs(metrics_dir, exist_ok=True)

# Prepare folder for structure related metrics
structure_metrics = os.path.join(
    metrics_dir,
    "structure",
)
os.makedirs(structure_metrics, exist_ok=True)

inner_similarity_jsonl_filename = os.path.join(structure_metrics, "inner_similarity.jsonl")
conditional_prefix_similarity_jsonl_filename = os.path.join(structure_metrics, "conditional_prefix_similarity.jsonl")
reference_similarity_jsonl_filename = os.path.join(structure_metrics, "reference_similarity.jsonl")
aggregated_similarities_json = os.path.join(structure_metrics, "aggregated_similarities.json")

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


def process_abc_file(abc_filename: str) -> tuple[
    str | None,
    tuple[str, SimilarityResult, SimilarityResult] | None,
    tuple[str, SimilarityResult | None, SimilarityResult | None] | None,
    tuple[str, SimilarityResult | None, SimilarityResult | None] | None,
]:
    try:
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
            subprocess.run(
                ["timidity", midi_file_path, "-Ow", "-o", wav_file_path, "-s", str(sample_rate)], check=False
            )
        elif midi_to_wav_converter == "FluidSynth":
            if sound_font:
                fs = FluidSynth(sound_font=os.path.join(sound_fonts_dir, sound_font), sample_rate=sample_rate)
            else:
                fs = FluidSynth(sound_font="/usr/share/sounds/sf2/default-GM.sf2", sample_rate=sample_rate)
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

        # Calculate structure similarity metrics
        vectorizer = MidiVectorizer()

        pitches_features, offsets_features = vectorizer.midi_or_score_to_notes_and_offsets_feature_vectors(
            midi_file_path
        )
        melody_inner_similarity_result = calculate_inner_similarity_of_music_vectors(
            pitches_features, cyclic_pitch_similarity
        )
        rhythm_inner_similarity_result = calculate_inner_similarity_of_music_vectors(
            offsets_features, rhythmic_similarity
        )

        # Reference similarity calculations
        melody_reference_similarity_result: SimilarityResult | None = None
        rhythm_reference_similarity_result: SimilarityResult | None = None
        if reference_midi_files_dir:
            # For ABC files, we might need to adjust the reference filename mapping
            # This assumes the reference files have similar naming pattern
            reference_filename = f"file_{idx}.mid"  # Adjust this pattern as needed
            reference_file_path = os.path.join(reference_midi_files_dir, reference_filename)
            if os.path.exists(reference_file_path):
                reference_pitches_features, reference_offsets_features = (
                    vectorizer.midi_or_score_to_notes_and_offsets_feature_vectors(reference_file_path)
                )
                melody_reference_similarity_result = calculate_reference_similarity_of_music_vectors(
                    pitches_features,
                    reference_pitches_features,
                    similarity_function=cyclic_pitch_similarity,
                    n_measures_to_skip=n_conditioned_measures,
                )
                rhythm_reference_similarity_result = calculate_reference_similarity_of_music_vectors(
                    offsets_features,
                    reference_offsets_features,
                    similarity_function=rhythmic_similarity,
                    n_measures_to_skip=n_conditioned_measures,
                )

        # Conditioned similarity calculations
        melody_conditioned_similarity_result: SimilarityResult | None = None
        rhythm_conditioned_similarity_result: SimilarityResult | None = None
        if n_conditioned_measures > 0:
            melody_conditioned_similarity_result = calculate_conditioned_similarity_of_music_vectors(
                pitches_features,
                conditioned_n_measures=n_conditioned_measures,
                similarity_function=cyclic_pitch_similarity,
            )
            rhythm_conditioned_similarity_result = calculate_conditioned_similarity_of_music_vectors(
                offsets_features,
                conditioned_n_measures=n_conditioned_measures,
                similarity_function=rhythmic_similarity,
            )

        return (
            os.path.abspath(wav_file_path),
            (midi_file_path, melody_inner_similarity_result, rhythm_inner_similarity_result),
            (midi_file_path, melody_reference_similarity_result, rhythm_reference_similarity_result),
            (midi_file_path, melody_conditioned_similarity_result, rhythm_conditioned_similarity_result),
        )
    except Exception as e:
        print(f"Error processing {abc_filename}: {e}")
        return None, None, None, None


if __name__ == "__main__":
    # Get a list of all ABC files in the input folder
    abc_files = sorted([f for f in os.listdir(abc_input_folder) if f.endswith(".abc")])

    # Process each ABC file using multiprocessing
    with multiprocessing.Pool() as pool:
        processing_results = list(tqdm(pool.imap(process_abc_file, abc_files), total=len(abc_files)))

    # Filter out None results and separate the results
    valid_results = [result for result in processing_results if result[0] is not None]

    # Write the collected WAV file paths to a JSONL file
    wav_paths = [result[0] for result in valid_results]
    with open(input_jsonl_filename, "w") as out_file:
        for path in wav_paths:
            json_line = json.dumps({"path": path})
            out_file.write(json_line + "\n")

    print(f"\nWAV file paths saved to {input_jsonl_filename}")

    # Save structure similarity metrics
    inner_similarity = [result[1] for result in valid_results if result[1] is not None]
    reference_similarity = [result[2] for result in valid_results if result[2] is not None]
    conditioned_similarity = [result[3] for result in valid_results if result[3] is not None]

    # Save inner similarity metrics
    with open(inner_similarity_jsonl_filename, "w") as out_file:
        for midi_file_path, melody_similarity, rhythm_sim in inner_similarity:
            json_line = json.dumps(
                {
                    "path": midi_file_path,
                    "melody": melody_similarity.model_dump(mode="json"),
                    "rhythm": rhythm_sim.model_dump(mode="json"),
                }
            )
            out_file.write(json_line + "\n")

    # Save reference similarity metrics
    with open(reference_similarity_jsonl_filename, "w") as out_file:
        for midi_file_path, melody_sim, rhythm_sim in reference_similarity:
            if melody_sim is None or rhythm_sim is None:
                continue
            json_line = json.dumps(
                {
                    "path": midi_file_path,
                    "melody": melody_sim.model_dump(mode="json"),
                    "rhythm": rhythm_sim.model_dump(mode="json"),
                }
            )
            out_file.write(json_line + "\n")

    # Save conditioned similarity metrics
    with open(conditional_prefix_similarity_jsonl_filename, "w") as out_file:
        for midi_file_path, melody_sim, rhythm_sim in conditioned_similarity:
            if melody_sim is None or rhythm_sim is None:
                continue
            json_line = json.dumps(
                {
                    "path": midi_file_path,
                    "melody": melody_sim.model_dump(mode="json"),
                    "rhythm": rhythm_sim.model_dump(mode="json"),
                }
            )
            out_file.write(json_line + "\n")

    # Aggregate structure similarity metrics
    def aggregate_similarity(jsonl_file: str) -> dict:
        if not os.path.exists(jsonl_file) or os.path.getsize(jsonl_file) == 0:
            return {"error": "No data available"}

        df = pd.read_json(jsonl_file, lines=True)
        if df.empty:
            return {"error": "No data available"}

        # Extract mean_best_similarities
        melody = df["melody"].apply(lambda x: x["mean_best_similarities"])
        rhythm = df["rhythm"].apply(lambda x: x["mean_best_similarities"])
        mean = pd.Series({"melody": melody.mean(), "rhythm": rhythm.mean()})
        se = pd.Series({"melody": melody.sem(ddof=1), "rhythm": rhythm.sem(ddof=1)})
        Z95 = NormalDist().inv_cdf(0.975)
        moe = se * Z95
        ci_lower = mean - moe
        ci_upper = mean + moe
        return {
            "mean": mean.to_dict(),
            "se": se.to_dict(),
            "moe": moe.to_dict(),
            "ci95_lower": ci_lower.to_dict(),
            "ci95_upper": ci_upper.to_dict(),
        }

    aggregated_similarities = {
        "inner": aggregate_similarity(inner_similarity_jsonl_filename),
        "reference": aggregate_similarity(reference_similarity_jsonl_filename) if reference_midi_files_dir else None,
        "conditioned": (
            aggregate_similarity(conditional_prefix_similarity_jsonl_filename) if n_conditioned_measures > 0 else None
        ),
    }

    with open(aggregated_similarities_json, "w") as f:
        json.dump(aggregated_similarities, f, indent=4)

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
