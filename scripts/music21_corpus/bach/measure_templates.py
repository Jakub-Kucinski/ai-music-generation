import json
from collections import Counter
from itertools import chain
from typing import cast

from music21 import corpus
from music21.common.types import OffsetQL
from music21.meter import TimeSignature
from music21.note import Note, Rest
from music21.stream import Measure, Score
from tqdm import tqdm

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.encodings.quantized_converter import (
    MidiQuantizedConverter,
)
from scripts.music21_corpus.bach.chorals_to_omit import CHORALS_TO_OMIT
from scripts.music21_corpus.bach.pydantic_models import (
    BachChord,
    BachMeasure,
    BachProgression,
)

settings = EncodingSetting(
    include_bars=True,
    include_rests=True,
    include_clef=True,
    include_key_signature=True,
    include_time_signature=True,
    include_offset_in_notes=True,
    include_offset_in_tuplets=True,
    joining_parts_strategy="Queue parallel measures",
    skip_measures_without_notes=False,
    notes_range=(21, 108),
    shortest_note_duration=16,
    longest_note_duration=2,
    allow_triplet_quarterLength=True,
    repeats_handling="Ignore",
    only_SATB_parts=True,
)
converter = MidiQuantizedConverter(settings)


nonchord_tones: int = 0
chord_tones: int = 0
bach_measures: list[BachMeasure] = []
bach_progressions: list[BachProgression] = []

# Search for Bach scores in the corpus.
metadata_bundle = corpus.search(composer="bach")
for metadata in tqdm(metadata_bundle):
    if not metadata.sourcePath.stem[:3].startswith("bwv") or metadata.sourcePath.stem in CHORALS_TO_OMIT:
        continue
    score: Score = cast(Score, metadata.parse())
    score = cast(Score, converter._quantize_stream(score))
    parts = converter.filter_allowed_parts(score)
    score = Score(parts)

    bach_chord_progression: list[BachChord] = []
    time_signature: TimeSignature | None = None
    four_parts_measures: list[list[Measure]] = []
    for part in score.parts:
        measures: list[Measure] = []
        for potential_measure in part:
            if type(potential_measure) is Measure:
                if time_signature is None:
                    time_signature = potential_measure.timeSignature
                measures.append(potential_measure)
        four_parts_measures.append(measures)
    if time_signature is None:
        time_signature = TimeSignature("4/4")
    time_signature_str = (
        f"{time_signature.numerator if time_signature.numerator else 4}"
        f"/{time_signature.denominator if time_signature.denominator else 4}"
    )

    if len(four_parts_measures) == 0:
        print(f"Got empty list of parts for file {metadata.sourcePath}")
        score.show("text", addEndTimes=True)
        continue
    n = len(four_parts_measures[0])
    if n != len(four_parts_measures[1]) and n != len(four_parts_measures[2]) and n != len(four_parts_measures[3]):
        raise RuntimeError(f"Got list of measures of different lengths in file {metadata.sourcePath}")

    for i in range(n):
        measure_stack: list[Measure] = [part_measure[i] for part_measure in four_parts_measures]

        parts_offsets: list[list[OffsetQL]] = []
        for measure in measure_stack:
            part_offsets: list[OffsetQL] = []
            for element in measure.getElementsByClass(Note):
                part_offsets.append(element.offset)
            parts_offsets.append(part_offsets)

        flat = chain.from_iterable(parts_offsets)
        counts = Counter(flat)
        bach_chords: list[BachChord] = []
        for offset, count in counts.items():
            # We treat only 3 or 4 simultaneously starting notes as chord.
            # We assume that 1 or 2 notes are only passing, neighbouring etc. notes
            if count <= 2:
                nonchord_tones += count
                continue
            chord_tones += count

            midi: list[int] = []
            is_start: list[bool] = []
            for part_offset, measure in zip(parts_offsets, measure_stack):
                last_note = cast(Note | Rest | None, measure.getElementAtOrBefore(offset, classList=[Note, Rest]))
                if last_note is None:
                    print(
                        f"Got no Note elements with getElementAtOrBefore for measure {i} and offset {offset}. "
                        f"File {metadata.sourcePath} Measure: {measure}"
                    )
                    measure.show("text", addEndTimes=True)
                    midi.append(0)
                    is_start.append(False)
                else:
                    if isinstance(last_note, Note):  # Note
                        midi.append(last_note.pitch.midi)
                        is_start.append(last_note.offset == offset)
                    else:  # Rest
                        midi.append(0)
                        is_start.append(last_note.offset == offset)
            chord = BachChord(
                offset=float(offset),
                midi=tuple(midi),  # type: ignore
                is_start=tuple(is_start),  # type: ignore
            )
            bach_chords.append(chord)
            bach_chord_progression.append(chord)
        if bach_chords:
            bach_measures.append(
                BachMeasure(
                    measure_duration=float(measure_stack[0].duration.quarterLength),
                    time_signature=time_signature_str,
                    bach_chords=bach_chords,
                )
            )
    bach_progression = BachProgression(bach_chords=bach_chord_progression)
    bach_progressions.append(bach_progression)


with open("scripts/music21_corpus/bach/stats/notes_counts.json", "w", encoding="utf-8") as f:
    notes_counts = {
        "chord_notes": chord_tones,
        "nonchord_notes": nonchord_tones,
    }
    json.dump(notes_counts, f, indent=4)


with open("scripts/music21_corpus/bach/stats/bach_measures.json", "w", encoding="utf-8") as f:
    json.dump([bach_measure.model_dump(mode="json") for bach_measure in bach_measures], f, indent=4)


with open("scripts/music21_corpus/bach/stats/bach_progression.json", "w", encoding="utf-8") as f:
    json.dump([progression.model_dump(mode="json") for progression in bach_progressions], f, indent=4)
