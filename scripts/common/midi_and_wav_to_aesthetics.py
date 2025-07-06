#!/usr/bin/env python3
import json
import multiprocessing
import os
import subprocess
from statistics import NormalDist

import pandas as pd
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

# Configuration
reference_midi_files_dir: str | None = None
n_conditioned_measures: int = 0

# Define paths
midi_input_folder = "data/04_generated/music21_bach_no_offsets_512_context/conditioned_4_bars/midi"
base_output_dir = "data/04_generated/music21_bach_no_offsets_512_context/conditioned_4_bars"
reference_midi_files_dir = "data/03_converted/music21_bach/validation/midi"
n_conditioned_measures = 4

# WAV files directory (assumed to already exist with files)
wav_output_dir = os.path.join(base_output_dir, "wav")

# Create subdirectory for metrics
metrics_dir = os.path.join(base_output_dir, "metrics")
os.makedirs(metrics_dir, exist_ok=True)

# Prepare folder for structure related metrics
structure_metrics = os.path.join(metrics_dir, "structure")
os.makedirs(structure_metrics, exist_ok=True)

inner_similarity_jsonl_filename = os.path.join(structure_metrics, "inner_similarity.jsonl")
conditional_prefix_similarity_jsonl_filename = os.path.join(structure_metrics, "conditional_prefix_similarity.jsonl")
reference_similarity_jsonl_filename = os.path.join(structure_metrics, "reference_similarity.jsonl")
aggregated_similarities_json = os.path.join(structure_metrics, "aggregated_similarities.json")

# Prepare folder for audiobox JSONL files (WAV paths and aesthetics)
audiobox_dir = os.path.join(metrics_dir, "audiobox_aesthetics")
os.makedirs(audiobox_dir, exist_ok=True)

input_jsonl_filename = os.path.join(audiobox_dir, "wav_paths.jsonl")
output_jsonl_filename = os.path.join(audiobox_dir, "aesthetics.jsonl")
output_aggregated_aesthetics = os.path.join(audiobox_dir, "aesthetics_aggregated.jsonl")


def process_midi_file(
    midi_filename: str,
) -> tuple[
    str,
    tuple[str, SimilarityResult, SimilarityResult],
    tuple[str, SimilarityResult | None, SimilarityResult | None],
    tuple[str, SimilarityResult | None, SimilarityResult | None],
]:
    midi_file_path = os.path.join(midi_input_folder, midi_filename)
    # Use the file name (without extension) as the index
    idx = os.path.splitext(midi_filename)[0]

    # Define corresponding WAV file path (assumed to already exist)
    wav_file_path = os.path.join(wav_output_dir, f"file_{idx}.wav")

    # Verify that the WAV file exists
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"Expected WAV file does not exist: {wav_file_path}")

    # Calculate similarities from MIDI file
    vectorizer = MidiVectorizer()

    pitches_features, offsets_features = vectorizer.midi_or_score_to_notes_and_offsets_feature_vectors(midi_file_path)
    melody_inner_similarity_result = calculate_inner_similarity_of_music_vectors(
        pitches_features, cyclic_pitch_similarity
    )
    rhythm_inner_similarity_result = calculate_inner_similarity_of_music_vectors(offsets_features, rhythmic_similarity)

    # Calculate reference similarity if reference directory is provided
    melody_reference_similarity_result: SimilarityResult | None = None
    rhythm_reference_similarity_result: SimilarityResult | None = None
    if reference_midi_files_dir:
        reference_filename = midi_filename.removeprefix("sample_")
        reference_file_path = os.path.join(reference_midi_files_dir, reference_filename)
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

    # Calculate conditioned similarity if n_conditioned_measures > 0
    melody_conditioned_similarity_result: SimilarityResult | None = None
    rhythm_conditioned_similarity_result: SimilarityResult | None = None
    if n_conditioned_measures > 0:
        melody_conditioned_similarity_result = calculate_conditioned_similarity_of_music_vectors(
            pitches_features, conditioned_n_measures=n_conditioned_measures, similarity_function=cyclic_pitch_similarity
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


if __name__ == "__main__":
    # Verify that required directories exist
    if not os.path.exists(midi_input_folder):
        raise FileNotFoundError(f"MIDI input folder does not exist: {midi_input_folder}")
    if not os.path.exists(wav_output_dir):
        raise FileNotFoundError(f"WAV output folder does not exist: {wav_output_dir}")

    # Get a list of all MIDI files in the input folder
    midi_files = sorted([f for f in os.listdir(midi_input_folder) if f.endswith(".mid")])
    print(f"Found {len(midi_files)} MIDI files")

    # Process each MIDI file using multiprocessing
    with multiprocessing.Pool() as pool:
        processing_results = list(tqdm(pool.imap(process_midi_file, midi_files), total=len(midi_files)))

    # Write the collected WAV file paths to a JSONL file
    wav_paths = [processing_result[0] for processing_result in processing_results]
    with open(input_jsonl_filename, "w") as out_file:
        for path in wav_paths:
            json_line = json.dumps({"path": path})
            out_file.write(json_line + "\n")

    print(f"\nWAV file paths saved to {input_jsonl_filename}")

    # Save metrics
    inner_similarity = [processing_result[1] for processing_result in processing_results]
    reference_similarity = [processing_result[2] for processing_result in processing_results]
    conditioned_similarity = [processing_result[3] for processing_result in processing_results]

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
        df = pd.read_json(jsonl_file, lines=True)
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
