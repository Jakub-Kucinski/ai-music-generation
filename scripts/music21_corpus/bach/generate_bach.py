import json
import random
from collections import defaultdict
from typing import cast

from music21.stream import Measure, Opus, Part, Score, Stream

from scripts.music21_corpus.bach.pydantic_models import (
    BachChord,
    BachMeasure,
    BachProgression,
)
from scripts.music21_corpus.bach.voices_ranges import ALTO, BASS, SOPRANO, TENOR


def select_time_signature_and_score_template() -> tuple[str, list[BachMeasure]]:
    with open("scripts/music21_corpus/bach/stats/measures_count.json", "w", encoding="utf-8") as f:
        measures_counts_dict: dict[str, list[int]] = json.load(f)
    time_signature_counts = {key: len(val) for key, val in measures_counts_dict.items()}
    items, weights = zip(*time_signature_counts.items())
    time_signature = cast(str, random.choices(items, weights=weights, k=1)[0])
    # n_measures = random.choice(measures_counts_dict[time_signature])

    with open("scripts/music21_corpus/bach/stats/score_templates.json", "w", encoding="utf-8") as f:
        score_templates_by_time_signature: dict[str, list[list[int]]] = json.load(f)
    #     measures_lengths_by_time_signature: dict[str, dict[str, int]] = json.load(f)
    # measures_length = measures_lengths_by_time_signature[time_signature]
    score_templates = score_templates_by_time_signature[time_signature]
    score_template = random.choice(score_templates)

    with open("scripts/music21_corpus/bach/stats/bach_measures.json", "w", encoding="utf-8") as f:
        list_of_bach_measures_dicts: list[dict] = json.load(f)
    list_of_bach_measures = [BachMeasure.model_validate(d) for d in list_of_bach_measures_dicts]
    list_of_bach_measures = [
        bach_measure for bach_measure in list_of_bach_measures if bach_measure.time_signature == time_signature
    ]

    measures_by_duration: dict[float, list[BachMeasure]] = defaultdict(list)
    for m in list_of_bach_measures:
        measures_by_duration[m.measure_duration].append(m)

    measures_template: list[BachMeasure] = []
    for measure_length in score_template:
        measures_template.append(random.choice(measures_by_duration[measure_length]))

    return time_signature, measures_template


def create_chord_progression_rules() -> (
    tuple[dict[tuple[int, int, int, int], list[BachChord]], dict[tuple[int, int, int, int], list[BachChord]]]
):
    with open("scripts/music21_corpus/bach/stats/bach_progression.json", "w", encoding="utf-8") as f:
        list_of_bach_chords_progressions_dicts: list[dict] = json.load(f)
    list_of_bach_chords_progressions = [
        BachProgression.model_validate(d) for d in list_of_bach_chords_progressions_dicts
    ]

    chord_to_possible_next_chords: dict[tuple[int, int, int, int], list[BachChord]] = defaultdict(list)
    chord_mod12_to_possible_next_chords: dict[tuple[int, int, int, int], list[BachChord]] = defaultdict(list)
    for bach_chord_progression in list_of_bach_chords_progressions:
        previous_chord: BachChord | None = None
        for bach_chord in bach_chord_progression.bach_chords:
            if previous_chord is None:
                previous_chord = bach_chord
                continue
            chord_to_possible_next_chords[previous_chord.midi].append(bach_chord)
            chord_mod12_to_possible_next_chords[previous_chord.midi_mod12()].append(bach_chord)
            previous_chord = bach_chord
    return chord_to_possible_next_chords, chord_mod12_to_possible_next_chords


def sample_chord_progression_for_template(
    time_signature: str,
    measure_templates: list[BachMeasure],
    chord_to_possible_next_chords: dict[tuple[int, int, int, int], list[BachChord]],
    chord_mod12_to_possible_next_chords: dict[tuple[int, int, int, int], list[BachChord]],
) -> list[BachChord]:
    chord_progression: list[BachChord] = []
    previous_chord: BachChord | None = None
    for measure_template in measure_templates:
        for offset_and_starting_notes in measure_template.bach_chords:
            if previous_chord is None:
                midis = random.choice(list(chord_to_possible_next_chords.keys()))
                previous_chord = BachChord(
                    offset=offset_and_starting_notes.offset,
                    midi=midis,
                    is_start=offset_and_starting_notes.is_start,
                )
                chord_progression.append(previous_chord)
                continue
            if previous_chord.midi in chord_to_possible_next_chords:
                previous_chord = random.choice(chord_to_possible_next_chords[previous_chord.midi])
                chord_progression.append(previous_chord)
            elif previous_chord.midi_mod12() in chord_mod12_to_possible_next_chords:
                previous_chord = random.choice(chord_mod12_to_possible_next_chords[previous_chord.midi_mod12()])
                chord_progression.append(previous_chord)
            else:
                print(f"No successor for {previous_chord} so choosing random")
                midis = random.choice(list(chord_to_possible_next_chords.keys()))
                previous_chord = BachChord(
                    offset=offset_and_starting_notes.offset,
                    midi=midis,
                    is_start=offset_and_starting_notes.is_start,
                )
                chord_progression.append(previous_chord)
    return chord_progression


def get_non_chord_note_frequency() -> float:
    with open("scripts/music21_corpus/bach/stats/notes_counts.json", "w", encoding="utf-8") as f:
        notes_counts: dict = json.load(f)
    return notes_counts["nonchord_notes"] / notes_counts["chord_notes"]  # type: ignore


def create_score(
    measure_templates: list[BachMeasure],
    chord_progression: list[BachChord],
    non_chord_note_probability: float,
) -> None:
    n_chords = len(chord_progression)
    i = 0

    parts = [Part() for _ in range(4)]

    # previous_chord: BachChord | None = None
    for measure_template in measure_templates:
        measures = [Measure() for _ in range(4)]
        for offset_and_starting_notes in measure_template.bach_chords:
            current_chord = chord_progression[i]
            if i + 1 < n_chords:
                next_chord = chord_progression[i + 1]
                can_add_non_chord_note_based_on_next = next_chord.is_start
            else:
                next_chord = None
                can_add_non_chord_note_based_on_next = (True, True, True, True)
            for i in range(4):
                if can_add_non_chord_note_based_on_next[i] and random.random() < non_chord_note_probability:
                    if next_chord is not None:
                        
            # previous_chord = current_chord
            i += 1
