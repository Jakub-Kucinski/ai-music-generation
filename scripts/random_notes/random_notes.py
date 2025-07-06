#!/usr/bin/env python3
import os
import random

from tqdm import tqdm

# Global configuration variables
MIN_NOTES = 1
MAX_NOTES = 4
MIN_TOTAL_NOTES = 64
OUTPUT_DIR = "data/04_generated/irishman/random_notes/abc"
NUM_FILES = 1000

# Single list of distinct notes
NOTES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "A,",
    "B,",
    "C,",
    "D,",
    "E,",
    "F,",
    "G,",
    "c'",
    "d'",
    "e'",
    "f'",
    "g'",
    "a'",
    "b'",
]

TIME_SIGNATURES = ["4/4", "3/4", "2/4", "2/2", "6/8", "7/8"]
KEY_SIGNATURES = ["C", "G", "D", "A", "F", "Bb", "Eb", "Am", "Em", "Dm"]


def generate_random_notes(num_notes: int, notes: list[str]) -> list[str]:
    """Generate a list of random notes with random durations appended (in the range 1-4) from the provided notes list"""
    result = []
    for _ in range(num_notes):
        note = random.choice(notes)
        duration = random.randint(1, 4)
        result.append(f"{note}{duration if duration > 1 else ""}")
    return result


def create_abc_content(i: int, measures: list[str], time_sig: str, key_sig: str) -> str:
    """Create an ABC file content with a header and a body that concatenates random measures.

    The header includes time signature, default note length, and key signature.
    """
    header = [f"X:{i}", "L:1/4", f"M:{time_sig}", f"K:{key_sig}"]
    body = " | ".join(measures)
    return "\n".join(header) + "\n" + body


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in tqdm(range(NUM_FILES)):
        selected_time_signature = random.choice(TIME_SIGNATURES)
        selected_key_signature = random.choice(KEY_SIGNATURES)

        measures = []
        total_notes = 0
        while total_notes < MIN_TOTAL_NOTES:
            measure_length = random.randint(MIN_NOTES, MAX_NOTES)
            measure_notes = generate_random_notes(measure_length, NOTES)
            measures.append(" ".join(measure_notes))
            total_notes += measure_length

        new_abc = create_abc_content(i + 1, measures, selected_time_signature, selected_key_signature)

        file_path = os.path.join(OUTPUT_DIR, f"file_{i+1}.abc")
        with open(file_path, "w") as f:
            f.write(new_abc)


if __name__ == "__main__":
    main()
