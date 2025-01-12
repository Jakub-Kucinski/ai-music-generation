from collections import defaultdict
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, Tuple, cast

import music21
import music21.meter
from music21 import Music21Object
from music21.chord import Chord
from music21.clef import Clef
from music21.common.types import OffsetQL, OffsetQLIn
from music21.key import KeySignature
from music21.meter import TimeSignature
from music21.note import GeneralNote, Note, NotRest, Rest
from music21.stream import Measure, Opus, Part, Score, Stream
from music21.stream.iterator import OffsetIterator
from pydantic import BaseModel

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.pydantic_models.instrument_types import InstrumentTypes


class TokenType(StrEnum):
    PITCH = "PITCH"
    DURATION = "DURATION"
    REST = "REST"
    BAR = "BAR"
    TIME_SHIFT = "TIME_SHIFT"
    TIME_SIGNATURE = "TIME_SIGNATURE"
    CLEF = "CLEF"
    KEY_SIGNATURE = "KEY_SIGNATURE"


class TupletModel(BaseModel):
    start_offset: float
    elements: list[Note | Chord | Rest]
    current_end: Fraction
    # math.gcd(numerator, denominator)


class MidiConverter:
    def __init__(
        self,
        settings: EncodingSetting = EncodingSetting(),
    ) -> None:
        self.settings = settings
        self.durations_range: Tuple[int, int] = (
            1,
            settings.longest_note_duration * settings.shortest_note_duration,
        )  # TODO: Ensure this is correct
        self.durations: list[str] = [  # TODO: Ensure this is correct
            f"d{i}" for i in range(self.durations_range[0], self.durations_range[1] + 1)
        ]
        self.pitches_range = settings.notes_range
        self.pitches: list[str] = [f"p{i}" for i in range(self.pitches_range[0], self.pitches_range[1] + 1)]
        self.rest: str = "rest"
        self.time_shift: str = "shift"
        self.bar: str = "bar"

        self.time_signatures: list[str] = (  # Most common time signatures
            []
            if not self.settings.include_time_signature
            else [
                f"time_signature_{i}"
                for i in [
                    "1/2,",
                    "2/2",
                    "3/2",
                    "1/4",
                    "2/4",
                    "3/4",
                    "4/4",
                    "5/4",
                    "6/4",
                    "7/4",
                    "1/8",
                    "2/8",
                    "3/8",
                    "4/8",
                    "5/8",
                    "6/8",
                    "7/8",
                    "8/8",
                    "9/8",
                    "10/8",
                    "11/8",
                    "12/8",
                ]
            ]
        )
        # [f"time_signature_{i}" for i in ["2/4", "3/4", "4/4", "2/2", "3/8", "6/8", "9/8", "12/8"]]
        self._clef_params = [  # Most common clefs
            ("G", 1, 0),
            ("G", 2, 0),
            ("G", 2, -1),
            ("G", 2, 1),
            ("G", 3, 0),
            ("C", 1, 0),
            ("C", 2, 0),
            ("C", 3, 0),
            ("C", 4, 0),
            ("C", 5, 0),
            ("F", 3, 0),
            ("F", 4, 0),
            ("F", 4, 1),
            ("F", 4, -1),
            ("F", 5, 0),
            ("TAB", 5, 0),
        ]
        self.clefs: list[str] = (
            []
            if not self.settings.include_clef
            else [f"clef_{sign}_{line}_{octaveChange}" for sign, line, octaveChange in self._clef_params]
        )
        self.key_signatures: list[str] = (
            [] if not self.settings.include_key_signature else [f"key_signature_{i}" for i in range(-7, 8)]
        )
        # TODO: Add all possible tokens for creating tokenizer
        self.all_possible_tokens, self.tokens_types = self._create_all_possible_tokens_list()

    def _create_all_possible_tokens_list(self) -> Tuple[list[str], list[TokenType]]:
        all_possible_tokens: list[str] = []
        tokens_types: list[TokenType] = []

        all_possible_tokens.append(self.time_shift)
        tokens_types.append(TokenType.TIME_SHIFT)

        if self.settings.include_bars:
            all_possible_tokens.append(self.bar)
            tokens_types.append(TokenType.BAR)
        if self.settings.include_rests:
            all_possible_tokens.append(self.rest)
            tokens_types.append(TokenType.REST)

        if self.settings.include_clef:
            all_possible_tokens.extend(self.clefs)
            tokens_types.extend([TokenType.CLEF] * len(self.clefs))
        if self.settings.include_key_signature:
            all_possible_tokens.extend(self.key_signatures)
            tokens_types.extend([TokenType.KEY_SIGNATURE] * len(self.key_signatures))
        if self.settings.include_time_signature:
            all_possible_tokens.extend(self.time_signatures)
            tokens_types.extend([TokenType.TIME_SIGNATURE] * len(self.time_signatures))

        all_possible_tokens.extend(self.durations)
        tokens_types.extend([TokenType.DURATION] * len(self.durations))
        all_possible_tokens.extend(self.pitches)
        tokens_types.extend([TokenType.PITCH] * len(self.pitches))

        if len(all_possible_tokens) != len(tokens_types):
            raise RuntimeError(
                f"Created all_possible_tokens and tokens_types of different lengths {len(all_possible_tokens)} != {len(tokens_types)}"  # noqa: E501
            )
        return all_possible_tokens, tokens_types

    def stream_to_texts(
        self,
        stream: Opus | Score | Part,
    ) -> dict[str, list[str]]:
        score_name_to_texts: dict[str, list[str]] = {}

        if type(stream) is Opus:
            for i, score in enumerate(stream.scores):
                single_score_name_to_texts = self.stream_to_texts(score)
                for score_name, texts in single_score_name_to_texts.items():
                    score_name_to_texts[f"{score_name}_{i}"] = texts
            return score_name_to_texts
        elif type(stream) is Score:
            score = stream
        elif type(stream) is Part:
            score = Score(stream)
        else:
            raise ValueError(f"Got stream of type {type(stream)}, but expected Opus | Score | Part")

        parts = self.filter_allowed_parts(score)
        for part in parts:
            updated_part = cast(Part, cast(Part, part.makeNotation()).makeRests(fillGaps=True))
            self._try_to_fix_broken_tuplets(updated_part)
            updated_part.makeTies(inPlace=True)

            last_clef: Clef | None = None
            last_time_signature: TimeSignature | None = None
            last_key_signature: KeySignature | None = None

            tuplets: list[TupletModel] = []

            offset_to_result_elements: dict[
                float, list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | TupletModel]
            ] = defaultdict(list)

            partOffsetIterator: OffsetIterator = OffsetIterator(updated_part)
            for elementGroup in partOffsetIterator:
                measure = None
                for element in elementGroup:
                    if type(element) is Measure:
                        measure = element
                        break
                if measure is None:
                    continue

                if isinstance(measure.offset, Fraction):
                    continue
                measureOffsetIterator: OffsetIterator = OffsetIterator(measure)

                for elements in measureOffsetIterator.getElementsByClass(
                    [Clef, KeySignature, TimeSignature, Note, Chord, Rest]
                ):
                    for element in elements:
                        element = cast(Clef | KeySignature | TimeSignature | Note | Chord | Rest, element)
                        if (
                            isinstance(element, Clef)
                            or isinstance(element, KeySignature)
                            or isinstance(element, TimeSignature)
                        ):
                            self._add_clef_key_or_time_signature_to_dict_if_changed(
                                last_clef,
                                last_key_signature,
                                last_time_signature,
                                offset_to_result_elements,
                                element,
                                measure.offset,
                            )
                            if isinstance(element, Clef):
                                last_clef = element
                            elif isinstance(element, KeySignature):
                                last_key_signature = element
                            elif isinstance(element, TimeSignature):
                                last_time_signature = element
                            continue

                        # Check if element is a part of a Tuplet
                        # (i.e. if element.duration.tuplets list is not empty)
                        # and if so assign it to the existing tuplet or create a new one
                        # Check if after appending to existing tuplet its end is no longer a Fraction
                        # (if its length is representable in binary system)
                        # At the end of this loop check if any tuplet current_end is lower then current offset
                        # If so, add this Tuplet (even if incompleted) to dict at its start_offset

                        # Handle tuplets - move this section to the _try_to_fix_broken_tuplets
                        # If offset is a fraction, then this element has to in an already created Tuplet
                        if isinstance(element.offset, Fraction):
                            pass
                        # If offset was not a fraction, but the
                        elif isinstance(element.duration.quarterLength, Fraction):
                            pass

        score_name_to_texts["score"] = []
        return score_name_to_texts

    def _try_to_fix_broken_tuplets(
        self,
        part: Part,
    ) -> None:
        raise NotImplementedError()

    def _add_clef_key_or_time_signature_to_dict_if_changed(
        self,
        last_clef: Clef | None,
        last_key_signature: KeySignature | None,
        last_time_signature: TimeSignature | None,
        offset_to_result_elements: dict[float, list[Any]],
        element: Clef | KeySignature | TimeSignature,
        measure_offset: float,
    ) -> None:
        if_element_changed = False
        if isinstance(element, Clef):
            if_element_changed = (
                last_clef is None
                or element.sign != last_clef.sign
                or element.line != last_clef.line
                or element.octaveChange != last_clef.octaveChange
            )
        elif isinstance(element, KeySignature):
            if_element_changed = last_key_signature is None or (
                isinstance(element.sharps, int)
                and isinstance(last_key_signature.sharps, int)
                and element.sharps != last_key_signature.sharps
            )
        elif isinstance(element, TimeSignature):
            if_element_changed = (
                last_time_signature is None
                or element.numerator != last_time_signature.numerator
                or element.denominator != last_time_signature.denominator
            )
        if if_element_changed:
            if isinstance(element.offset, Fraction):
                offset_to_result_elements[measure_offset].append(element)
            else:
                offset_to_result_elements[element.offset].append(element)

    def filepath_to_texts(
        self,
        midi_path: Path,
    ) -> dict[str, list[str]]:
        stream = music21.converter.parseFile(midi_path)
        score_name_to_texts = self.stream_to_texts(stream)
        return score_name_to_texts

    def filter_allowed_parts(self, score: Score) -> list[Part]:
        accepted_parts = []
        for part in score.parts:
            if bool(part.recurse().getElementsByClass(Note)) or bool(part.recurse().getElementsByClass(Chord)):
                if self.is_allowed_part_instrument(part):
                    accepted_parts.append(part)
        return accepted_parts

    def is_allowed_part_instrument(self, part: Part) -> bool:
        n_instruments = 0
        for instrument in part.getInstruments():
            n_instruments += 1
            if instrument.midiProgram is not None:
                is_allowed = False
                for allowed_instrument in self.settings.allowed_instruments:
                    if instrument.midiProgram in allowed_instrument.value:
                        is_allowed = True
                if not is_allowed:
                    return False
            elif instrument.midiChannel is not None:
                if instrument.midiChannel == 9 and InstrumentTypes.PERCUSSIVE in self.settings.allowed_instruments:
                    pass
                else:
                    return False
            else:
                return False
        # If loop ended and there was at least 1 instrument, we mark this part as allowed
        return n_instruments > 0
