import json
from collections import Counter, defaultdict
from typing import cast

from music21 import corpus
from music21.meter import TimeSignature
from music21.stream import Measure, Score
from music21.stream.iterator import OffsetIterator
from tqdm import tqdm

from scripts.music21_corpus.bach.chorals_to_omit import CHORALS_TO_OMIT

# Search for Bach scores in the corpus.
metadata_bundle = corpus.search(composer="bach")

time_signature_to_score_templates: dict[str, list[list[int]]] = defaultdict(list)
for metadata in tqdm(metadata_bundle):
    if not metadata.sourcePath.stem[:3].startswith("bwv") or metadata.sourcePath.stem in CHORALS_TO_OMIT:
        continue
    score: Score = cast(Score, metadata.parse())
    part = score.parts[0]
    time_signature: TimeSignature | None = None
    measures_lengths: list[int] = []
    partOffsetIterator: OffsetIterator = OffsetIterator(part)
    for elementGroup in partOffsetIterator:
        measure = None
        for possible_measure in elementGroup:
            if type(possible_measure) is Measure:
                measure = possible_measure
                break
        if measure is None:
            continue
        if time_signature is None:
            time_signature = measure.timeSignature
        measures_lengths.append(int(measure.duration.quarterLength))
    if time_signature is None:
        time_signature = TimeSignature("4/4")
    time_signature_fraction = (
        time_signature.numerator if time_signature.numerator else 4,
        time_signature.denominator if time_signature.denominator else 4,
    )
    time_signature_to_score_templates[f"{time_signature_fraction[0]}/{time_signature_fraction[1]}"].append(
        measures_lengths
    )

with open("scripts/music21_corpus/bach/stats/score_templates.json", "w", encoding="utf-8") as f:
    json.dump(time_signature_to_score_templates, f, indent=4)

counts_by_key: dict[str, Counter[int]] = {
    key: Counter(val for sublist in matrix for val in sublist)  # matrix is a list[list[float]]  # iterate each float
    for key, matrix in time_signature_to_score_templates.items()
}

# Example of accessing the counts:
for key, counter in counts_by_key.items():
    print(f"Key {key}:")
    for float_val, cnt in counter.items():
        print(f"  {float_val!r} → {cnt}")

lengths_to_save = {
    key: {measure_length: count for measure_length, count in counter.items()} for key, counter in counts_by_key.items()
}

with open("scripts/music21_corpus/bach/stats/measure_length_counts.json", "w", encoding="utf-8") as f:
    json.dump(lengths_to_save, f, indent=4)


counts_to_save = {
    key: [len(score_template) for score_template in score_templates]
    for key, score_templates in time_signature_to_score_templates.items()
}

with open("scripts/music21_corpus/bach/stats/measures_count.json", "w", encoding="utf-8") as f:
    json.dump(counts_to_save, f, indent=4)
