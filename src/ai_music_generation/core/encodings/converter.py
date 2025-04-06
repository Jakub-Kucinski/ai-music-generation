from collections import defaultdict
from enum import StrEnum
from fractions import Fraction
from math import gcd, prod
from pathlib import Path
from typing import Any, Tuple, cast

import music21
import music21.meter
from devtools import pprint
from loguru import logger
from music21 import Music21Object
from music21.chord import Chord
from music21.clef import Clef
from music21.common.numberTools import opFrac
from music21.common.types import OffsetQL, OffsetQLIn
from music21.duration import Duration, Tuplet
from music21.key import KeySignature
from music21.meter import TimeSignature
from music21.note import GeneralNote, Note, NotRest, Rest
from music21.stream import Measure, Opus, Part, Score, Stream
from music21.stream.iterator import OffsetIterator
from pydantic import BaseModel, ConfigDict

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
    start_offset: OffsetQL
    elements: list[Note | Chord | Rest]
    current_end_offset: OffsetQL
    # math.gcd(numerator, denominator)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def normal_duration(self) -> OffsetQL:
        return cast(OffsetQL, opFrac(self.current_end_offset - self.start_offset))

    @property
    def actual_duration(self) -> OffsetQL:
        act_duration: OffsetQL = 0.0
        for element in self.elements:
            act_duration = cast(OffsetQL, opFrac(act_duration + element.duration.quarterLengthNoTuplets))
        return act_duration


class BarModel(BaseModel):
    bar_duration_quarterLength: OffsetQL
    real_duration_quarterLength: OffsetQL

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
        self.bar: str = "|"
        self.tuplet_start: str = "tuplet_start"
        self.tuplet_end: str = "tuplet_end"

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
        file_name: str,
    ) -> dict[str, list[str]]:
        score_name_to_texts: dict[str, list[str]] = {}
        # TODO:
        # Maybe add saving info about instrument
        if type(stream) is Opus:
            for i, score in enumerate(stream.scores):
                single_score_name_to_texts = self.stream_to_texts(score, f"{file_name}_{i}")
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
            TODO_1, TODO_2 = self._try_to_fix_broken_tuplets(updated_part)
            updated_part.makeTies(inPlace=True)

            last_clef: Clef | None = None
            last_time_signature: TimeSignature | None = None
            last_key_signature: KeySignature | None = None

            uncompleted_tuplets: list[TupletModel] = []

            offset_to_result_elements: dict[
                float | Fraction,
                list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | TupletModel | BarModel],
            ] = defaultdict(list)

            partOffsetIterator: OffsetIterator = OffsetIterator(updated_part)
            for elementGroup in partOffsetIterator:
                measure = None
                for possible_measure in elementGroup:
                    if type(possible_measure) is Measure:
                        measure = possible_measure
                        break
                if measure is None:
                    continue

                if measure.hasVoices():
                    measure.flattenUnnecessaryVoices(force=False, inPlace=True)
                    if measure.hasVoices():
                        measure = cast(Measure, measure.chordify())

                if isinstance(measure.offset, Fraction):
                    continue

                offset_to_result_elements[measure.offset].append(
                    BarModel(
                        bar_duration_quarterLength=measure.barDuration.quarterLength,
                        real_duration_quarterLength=measure.duration.quarterLength,
                    )
                )
                measureOffsetIterator: OffsetIterator = OffsetIterator(measure)

                for elements in measureOffsetIterator.getElementsByClass(
                    [Clef, KeySignature, TimeSignature, Note, Chord, Rest]
                ):
                    current_loop_offset = measure.offset
                    for element in elements:
                        element = cast(Clef | KeySignature | TimeSignature | Note | Chord | Rest, element)
                        current_loop_offset = element.offset
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

                        # If offset is a fraction, then this element has to be added to an already created Tuplet
                        if isinstance(element.offset, Fraction):
                            for tuplet in uncompleted_tuplets:
                                if opFrac(tuplet.current_end_offset) == opFrac(measure.offset + element.offset):
                                    tuplet.elements.append(element)
                                    tuplet.current_end_offset = cast(
                                        OffsetQL, opFrac(tuplet.current_end_offset + element.duration.quarterLength)
                                    )
                                    break
                        # If offset was not a fraction, but the duration is a fraction then
                        # it belongs to a tuplet (probably beginning of a new one)
                        elif isinstance(element.duration.quarterLength, Fraction):
                            uncompleted_tuplets.append(
                                TupletModel(
                                    start_offset=cast(OffsetQL, opFrac(measure.offset + element.offset)),
                                    elements=[element],
                                    current_end_offset=cast(
                                        OffsetQL,
                                        opFrac(
                                            cast(OffsetQL, opFrac(measure.offset + element.offset))
                                            + element.duration.quarterLength
                                        ),
                                    ),
                                )
                            )
                        # Normal case
                        else:
                            offset_to_result_elements[cast(OffsetQL, opFrac(measure.offset + element.offset))].append(
                                element
                            )

                    # At the end of this loop check if any tuplet current_end_offset is lower then current offset
                    # If so, add this Tuplet (even if incompleted) to dict at its start_offset
                    _uncompleted_tuplets: list[TupletModel] = []
                    for tuplet in uncompleted_tuplets:
                        if tuplet.current_end_offset <= cast(OffsetQL, opFrac(measure.offset + current_loop_offset)):
                            offset_to_result_elements[tuplet.start_offset].append(tuplet)
                        else:
                            _uncompleted_tuplets.append(tuplet)
                    uncompleted_tuplets = _uncompleted_tuplets

            text = self._convert_offset_dict_to_text(offset_to_result_elements=offset_to_result_elements)
            if file_name not in score_name_to_texts:
                score_name_to_texts[file_name] = [text]
            else:
                score_name_to_texts[file_name].append(text)
        return score_name_to_texts

    def _convert_offset_dict_to_text(
        self,
        offset_to_result_elements: dict[
            float | Fraction,
            list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | TupletModel | BarModel],
        ],
    ) -> str:
        tokens: list[str] = []
        offsets = sorted(offset_to_result_elements.keys())
        is_first_bar: bool = True
        element = None
        measure_offset: float = 0
        for offset in offsets:
            if isinstance(offset, Fraction):
                continue
            elements = offset_to_result_elements[offset]

            clef = next((element for element in elements if isinstance(element, Clef)), None)
            if clef is not None and self.settings.include_clef:
                tokens.append(f"clef_{clef.sign}_{clef.line}_{clef.octaveChange}")
            key_signature = next((element for element in elements if isinstance(element, KeySignature)), None)
            if key_signature is not None and self.settings.include_key_signature:
                tokens.append(f"key_signature_{key_signature.sharps}")
            time_signature = next((element for element in elements if isinstance(element, TimeSignature)), None)
            if time_signature is not None and self.settings.include_time_signature:
                tokens.append(f"time_signature_{time_signature.numerator}/{time_signature.denominator}")

            bar = next((element for element in elements if isinstance(element, BarModel)), None)
            if bar is not None:
                if self.settings.include_bars:
                    if is_first_bar:
                        is_first_bar = False
                    else:
                        offset_int = self.duration_or_offset_to_int_enc(opFrac(offset - measure_offset))
                        tokens.append(f"o{offset_int}")
                        tokens.append(self.bar)
                measure_offset = offset

            in_bar_offset = cast(OffsetQL, opFrac(offset - measure_offset))
            if self.settings.include_offset and (
                any(isinstance(element, (Note, Chord)) for element in elements)
                or (self.settings.include_rests and any(isinstance(element, Rest) for element in elements))
            ):
                offset_int = self.duration_or_offset_to_int_enc(in_bar_offset)
                tokens.append(f"o{offset_int}")

            for element in elements:
                if isinstance(element, Note):
                    tokens.append(f"p{element.pitch.midi}")
                    tokens.append(f"d{self.duration_or_offset_to_int_enc(element.duration.quarterLength)}")
                elif isinstance(element, Rest) and self.settings.include_rests:
                    tokens.append(self.rest)
                    tokens.append(f"d{self.duration_or_offset_to_int_enc(element.duration.quarterLength)}")
                elif isinstance(element, Chord):
                    for pitch in element.pitches:
                        tokens.append(f"p{pitch.midi}")
                    tokens.append(f"d{self.duration_or_offset_to_int_enc(element.duration.quarterLength)}")
                elif isinstance(element, TupletModel):
                    tokens.append(self.tuplet_start)
                    tokens.append(f"d{self.duration_or_offset_to_int_enc(element.normal_duration)}")
                    tokens.append(f"d{self.duration_or_offset_to_int_enc(element.actual_duration)}")
                    in_tuplet_offset: OffsetQL = 0.0
                    for note_or_rest in element.elements:
                        if self.settings.include_offset_in_tuplets:
                            tokens.append(f"o{self.duration_or_offset_to_int_enc(in_tuplet_offset)}")
                        in_tuplet_offset = cast(
                            OffsetQL, opFrac(in_tuplet_offset + note_or_rest.duration.quarterLengthNoTuplets)
                        )
                        if isinstance(note_or_rest, Note):
                            tokens.append(f"p{note_or_rest.pitch.midi}")
                        elif isinstance(note_or_rest, Rest) and self.settings.include_rests:
                            tokens.append(self.rest)
                        elif isinstance(note_or_rest, Chord):
                            for pitch in note_or_rest.pitches:
                                tokens.append(f"p{pitch.midi}")
                        else:
                            continue
                        tokens.append(
                            f"d{self.duration_or_offset_to_int_enc(note_or_rest.duration.quarterLengthNoTuplets)}"
                        )
                    tokens.append(self.tuplet_end)
        # Add last bar
        if self.settings.include_bars and element is not None:
            if isinstance(element, TupletModel):
                if self.settings.include_offset:
                    offset_int = self.duration_or_offset_to_int_enc(opFrac(element.current_end_offset - measure_offset))
                    tokens.append(f"o{offset_int}")
                tokens.append(self.bar)
            elif isinstance(element, BarModel):
                pass
            else:
                if element.duration.quarterLength > 0 and self.settings.include_offset:
                    offset_int = self.duration_or_offset_to_int_enc(
                        opFrac(element.offset + element.duration.quarterLength)
                    )
                    tokens.append(f"o{offset_int}")
                tokens.append(self.bar)
        return " ".join(tokens)

    def _try_to_fix_broken_tuplets(
        self,
        part: Part,
    ) -> Tuple[list[TupletModel], list[TupletModel]]:
        uncompleted_tuplets: list[TupletModel] = []
        completed_tuplets: list[TupletModel] = []
        partOffsetIterator: OffsetIterator = OffsetIterator(part)
        for elementGroup in partOffsetIterator:
            measure = None
            for element in elementGroup:
                if type(element) is Measure:
                    measure = element
                    break
            if measure is None:
                continue

            measureOffsetIterator: OffsetIterator = OffsetIterator(measure)

            for elements in measureOffsetIterator.getElementsByClass([Note, Chord, Rest]):
                for element in elements:
                    element = cast(Note | Chord | Rest, element)
                    # Check if element is a part of a Tuplet
                    # (i.e. if element.duration.tuplets list is not empty)
                    # and if so assign it to the existing tuplet or create a new one
                    # Check if after appending to existing tuplet its end is no longer a Fraction
                    # (if its length is representable in binary system)
                    # At the end of this loop check if any tuplet current_end_offset is lower then current offset
                    # If so, add this Tuplet (even if incompleted) to dict at its start_offset

                    # If offset is a fraction, then this element has to be added to an already created Tuplet
                    if isinstance(element.offset, Fraction):
                        for tuplet in uncompleted_tuplets:
                            if opFrac(tuplet.current_end_offset) == opFrac(measure.offset + element.offset):
                                tuplet.elements.append(element)
                                break

                    # If offset was not a fraction, but the duration is a fraction then
                    # it belongs to a tuplet (probably beginning of a new one)
                    elif isinstance(element.duration.quarterLength, Fraction):
                        uncompleted_tuplets.append(
                            TupletModel(
                                start_offset=cast(OffsetQL, opFrac(measure.offset + element.offset)),
                                elements=[element],
                                current_end_offset=cast(
                                    OffsetQL,
                                    opFrac(
                                        cast(OffsetQL, opFrac(measure.offset + element.offset))
                                        + element.duration.quarterLength
                                    ),
                                ),
                            )
                        )
            new_uncompleted_tuplets: list[TupletModel] = []
            for tuplet in uncompleted_tuplets:
                if isinstance(tuplet.current_end_offset, Fraction):
                    new_uncompleted_tuplets.append(tuplet)
                else:
                    completed_tuplets.append(tuplet)
            uncompleted_tuplets = new_uncompleted_tuplets

        for completed_tuplet in completed_tuplets:
            # Iterate over all elements of tuplet and over all tuplets in these elements
            # to get all tuplets multipliers / ratios
            # Then iterate again over all elements and add missing tuplets ratios and fix durations accordingly
            music21_tuplets: list[Tuplet] = []
            current_ratios: list[Tuple[int, int]] = []
            for element in completed_tuplet.elements:
                current_actual = 1
                current_normal = 1
                for music21_tuplet in element.duration.tuplets:
                    is_known = False
                    current_actual *= music21_tuplet.numberNotesActual
                    current_normal *= music21_tuplet.numberNotesNormal
                    for known_music21_tuplet in music21_tuplets:
                        if (
                            music21_tuplet.numberNotesActual == known_music21_tuplet.numberNotesActual
                            and music21_tuplet.numberNotesNormal == known_music21_tuplet.numberNotesNormal
                        ):
                            is_known = True
                            break
                    if not is_known:
                        music21_tuplets.append(music21_tuplet)
                current_ratios.append((current_actual, current_normal))

            total_actual = prod(ratio[0] for ratio in current_ratios)
            total_normal = prod(ratio[1] for ratio in current_ratios)
            divisor = gcd(total_actual, total_normal)
            total_actual //= divisor
            total_normal //= divisor
            # Update durations of all notes to the common tuplet ratio (total_actual, total_normal)
            for element, (current_actual, current_normal) in zip(
                completed_tuplet.elements, current_ratios, strict=True
            ):
                # TODO: Add updating durations of tuplet elements by changing duration and adding tuplet
                new_duration = music21.duration.Duration(
                    opFrac(
                        element.duration.quarterLength
                        * cast(OffsetQL, opFrac((current_normal * total_actual) / (current_actual * total_normal)))
                    )
                )
                element.duration = new_duration
                for music21_tuplet in music21_tuplets:
                    element.duration.appendTuplet(music21_tuplet)
        return completed_tuplets, uncompleted_tuplets

    def _add_clef_key_or_time_signature_to_dict_if_changed(
        self,
        last_clef: Clef | None,
        last_key_signature: KeySignature | None,
        last_time_signature: TimeSignature | None,
        offset_to_result_elements: dict[
            float | Fraction, list[Clef | KeySignature | TimeSignature | Note | Chord | Rest | TupletModel | BarModel]
        ],
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
                offset_to_result_elements[cast(OffsetQL, opFrac(measure_offset + element.offset))].append(element)

    def filepath_to_texts(
        self,
        midi_path: Path,
    ) -> dict[str, list[str]]:
        stream = music21.converter.parseFile(midi_path)
        score_name_to_texts = self.stream_to_texts(stream, midi_path.name)
        return score_name_to_texts

    def filter_allowed_parts(self, score: Score) -> list[Part]:
        accepted_parts = []
        for part in score.parts:
            if bool(part.recurse().getElementsByClass(Note)) or bool(part.recurse().getElementsByClass(Chord)):
                if not self.settings.allowed_instruments:
                    accepted_parts.append(part)
                elif self.is_allowed_part_instrument(part):
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

    def duration_or_offset_to_int_enc(self, quarterLength: float | Fraction | None) -> int:
        if quarterLength is None:
            raise ValueError(f"Got quarterLength which is None {quarterLength}")
        duration_as_int = quarterLength * (self.settings.shortest_note_duration / 4)
        if not duration_as_int.is_integer():
            error_message = (
                f"Encountered note whose duration {quarterLength / 4} couldn't be "
                "represented as integer multiple of "
                f"self.settings.shortest_note_duration {self.settings.shortest_note_duration}"
            )
            logger.warning(error_message)
            if self.settings.raise_duration_errors:
                raise ValueError(error_message)
            return max(1, int(duration_as_int))
        if quarterLength > self.settings.longest_note_duration * 4:
            error_message = (
                f"Encountered note whose duration ({quarterLength / 4} in whole notes, "
                f"{quarterLength} in quarterLength) is bigger than "
                f"self.settings.longest_note_duration {self.settings.longest_note_duration}"
            )
            logger.warning(error_message)
            if self.settings.raise_duration_errors:
                raise ValueError(error_message)
            return self.settings.longest_note_duration * 4
        return int(duration_as_int)

    def int_enc_to_quarterLength(self, int_enc: int) -> float:
        return int_enc * 4 / self.settings.shortest_note_duration
