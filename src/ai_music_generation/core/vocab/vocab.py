from collections import defaultdict
from enum import StrEnum
from typing import Tuple

from devtools import pformat, pprint
from loguru import logger

from ai_music_generation.core.encodings.encoding_settings import EncodingSetting
from ai_music_generation.core.pydantic_models.musical_notation import (
    BarModel,
    ClefModel,
    KeySignatureModel,
    NoteModel,
    RestModel,
    TimeShiftModel,
    TimeSignatureModel,
)


class TokenType(StrEnum):
    PITCH = "PITCH"
    DURATION = "DURATION"
    REST = "REST"
    BAR = "BAR"
    TIME_SHIFT = "TIME_SHIFT"
    TIME_SIGNATURE = "TIME_SIGNATURE"
    CLEF = "CLEF"
    KEY_SIGNATURE = "KEY_SIGNATURE"


class Vocab:
    def __init__(
        self,
        encoding_settings: EncodingSetting = EncodingSetting(),
    ) -> None:
        self.encoding_settings = encoding_settings
        self.durations_range: Tuple[int, int] = (
            1,
            encoding_settings.longest_note_duration * encoding_settings.shortest_note_duration,
        )  # TODO: Ensure this is correct
        self.durations: list[str] = [  # TODO: Ensure this is correct
            f"d{i}" for i in range(self.durations_range[0], self.durations_range[1] + 1)
        ]
        self.pitches_range = encoding_settings.notes_range
        self.pitches: list[str] = [f"p{i}" for i in range(self.pitches_range[0], self.pitches_range[1] + 1)]
        self.rest: str = "rest"
        self.time_shift: str = "shift"
        self.bar: str = "bar"

        self.time_signatures: list[str] = (  # Most common time signatures
            []
            if not self.encoding_settings.include_time_signature
            else [f"time_signature_{i}" for i in ["2/4", "3/4", "4/4", "2/2", "6/8", "9/8", "12/8"]]
        )
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
            if not self.encoding_settings.include_clef
            else [f"clef_{sign}_{line}_{octaveChange}" for sign, line, octaveChange in self._clef_params]
        )
        self.key_signatures: list[str] = (
            [] if not self.encoding_settings.include_key_signature else [f"key_signature_{i}" for i in range(-7, 8)]
        )
        # TODO: Add all possible tokens for creating tokenizer
        self.all_possible_tokens, self.tokens_types = self._create_all_possible_tokens_list()

    def _create_all_possible_tokens_list(self) -> Tuple[list[str], list[TokenType]]:
        all_possible_tokens: list[str] = []
        tokens_types: list[TokenType] = []

        all_possible_tokens.append(self.time_shift)
        tokens_types.append(TokenType.TIME_SHIFT)

        if self.encoding_settings.include_bars:
            all_possible_tokens.append(self.bar)
            tokens_types.append(TokenType.BAR)
        if self.encoding_settings.include_rests:
            all_possible_tokens.append(self.rest)
            tokens_types.append(TokenType.REST)

        if self.encoding_settings.include_clef:
            all_possible_tokens.extend(self.clefs)
            tokens_types.extend([TokenType.CLEF] * len(self.clefs))
        if self.encoding_settings.include_key_signature:
            all_possible_tokens.extend(self.key_signatures)
            tokens_types.extend([TokenType.KEY_SIGNATURE] * len(self.key_signatures))
        if self.encoding_settings.include_time_signature:
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

    def offset_mapping_to_text(
        self,
        offset_to_notes: dict[int, list[NoteModel | RestModel | BarModel]],
        clef: ClefModel | None,
        key_signature: KeySignatureModel | None,
        time_signature: TimeSignatureModel | None,
    ) -> str:
        if self.encoding_settings.include_clef and clef is None:
            raise ValueError(f"include_clef was {self.encoding_settings.include_clef}, but got clef {clef}")
        if self.encoding_settings.include_key_signature and key_signature is None:
            raise ValueError(
                f"include_key_signature was {self.encoding_settings.include_key_signature}, but got key_signature {key_signature}"  # noqa: E501
            )
        if self.encoding_settings.include_time_signature and time_signature is None:
            raise ValueError(
                f"include_time_signature was {self.encoding_settings.include_time_signature}, but got time_signature {time_signature}"  # noqa: E501
            )

        tokens: list[str] = []
        if clef is not None:
            clef_str = f"clef_{clef.sign}_{clef.line}_{clef.octaveChange}"
            if clef_str not in self.clefs:
                raise ValueError(f"Clef {clef_str} was not found in self.clefs {self.clefs}")
            tokens.append(clef_str)
        if key_signature is not None:
            key_signature_str = f"key_signature_{key_signature.sharps}"
            if key_signature_str not in self.key_signatures:
                raise ValueError(
                    f"key_signature {key_signature_str} was not found in self.key_signatures {self.key_signatures}"
                )
            tokens.append(key_signature_str)
        if time_signature is not None:
            time_signature_str = f"time_signature_{time_signature.numerator}/{time_signature.denominator}"
            if time_signature_str not in self.time_signatures:
                raise ValueError(
                    f"time_signature {time_signature_str} was not found in self.time_signatures {self.time_signatures}"
                )
            tokens.append(time_signature_str)

        last_offset: int = 0
        for offset in sorted(offset_to_notes.keys()):
            tokens_to_add: list[str] = []
            if offset != 0:
                tokens.append(self.time_shift)
                tokens.append(f"d{offset - last_offset}")
            save_bar_at_current_offset: bool = False
            for note in offset_to_notes[offset]:
                if isinstance(note, NoteModel):
                    tokens_to_add.append(f"p{note.pitch}")
                    tokens_to_add.append(f"d{note.duration}")
                elif isinstance(note, RestModel) and self.encoding_settings.include_rests:
                    tokens_to_add.append(self.rest)
                    tokens_to_add.append(f"d{note.duration}")
                elif isinstance(note, BarModel):
                    save_bar_at_current_offset = True
            if self.encoding_settings.include_bars and save_bar_at_current_offset:
                tokens_to_add = [self.bar] + tokens_to_add
            tokens.extend(tokens_to_add)
            last_offset = offset
        pprint(offset_to_notes)
        print(tokens)
        self._check_if_tokens_in_vocab(tokens)
        return " ".join(tokens)

    def _check_if_tokens_in_vocab(self, tokens: list[str]) -> bool:
        disallowed_tokens: list[str] = []
        for token in tokens:
            if token not in self.all_possible_tokens:
                disallowed_tokens.append(token)
        if len(disallowed_tokens) > 0:
            raise ValueError(
                f"Provided tokens contained values that were not allowed by encoding settings {pformat(self.encoding_settings)}\n"  # noqa: E501
                f"disallowed_tokens: {disallowed_tokens}"
            )
        return True

    def _get_token_type(self, token: str) -> TokenType:
        index = self.all_possible_tokens.index(token)
        return self.tokens_types[index]

    def _clef_token_to_model(self, token: str) -> ClefModel:
        index = self.clefs.index(token)
        params = self._clef_params[index]
        clef = ClefModel(
            sign=params[0],
            line=params[1],
            octaveChange=params[2],
        )
        return clef

    def _key_signature_token_to_model(self, token: str) -> KeySignatureModel:
        _ = self.key_signatures.index(token)
        n_sharps = token.split("_")[-1]
        return KeySignatureModel(sharps=int(n_sharps))

    def _time_signature_token_to_model(self, token: str) -> TimeSignatureModel:
        _ = self.time_signatures.index(token)
        fraction = token.split("_")[-1]
        numerator, denominator = fraction.split("/", maxsplit=1)
        return TimeSignatureModel(numerator=int(numerator), denominator=int(denominator))

    def _duration_token_to_int(self, duration_token: str) -> int:
        _ = self.durations.index(duration_token)
        duration = int(duration_token[1:])
        return duration

    def _note_tokens_to_model(self, pitch_token: str, duration_token: str) -> NoteModel:
        _ = self.pitches.index(pitch_token)
        pitch_midi_number = int(pitch_token[1:])
        duration = self._duration_token_to_int(duration_token)
        return NoteModel(pitch=pitch_midi_number, duration=duration)

    def _rest_token_to_model(self, rest_token: str, duration_token: str) -> RestModel:
        assert rest_token == self.rest
        duration = self._duration_token_to_int(duration_token)
        return RestModel(duration=duration)

    def _time_shift_token_to_model(self, time_shift_token: str, duration_token: str) -> TimeShiftModel:
        assert time_shift_token == self.time_shift
        duration = self._duration_token_to_int(duration_token)
        return TimeShiftModel(duration=duration)

    def _bar_model_token_to_model(self, bar_model_token: str) -> BarModel:
        assert bar_model_token == self.bar
        return BarModel()

    def text_to_offset_mapping(self, text: str) -> Tuple[
        dict[int, list[NoteModel | RestModel | BarModel]],
        ClefModel | None,
        KeySignatureModel | None,
        TimeSignatureModel | None,
        list[str],
    ]:
        clef: ClefModel | None = None
        key_signature: KeySignatureModel | None = None
        time_signature: TimeSignatureModel | None = None

        decoding_errors: list[str] = []

        offset_to_notes: dict[int, list[NoteModel | RestModel | BarModel]] = defaultdict(list)

        tokens = text.split(" ")
        if self.encoding_settings.include_clef:
            token = tokens[0]
            if self._get_token_type(token) is not TokenType.CLEF:
                t = f"Expected clef, but got {token}"
                logger.warning(t)
                decoding_errors.append(t)
            else:
                clef = self._clef_token_to_model(token)
                tokens = tokens[1:]

        if self.encoding_settings.include_key_signature:
            token = tokens[0]
            if self._get_token_type(token) is not TokenType.KEY_SIGNATURE:
                t = f"Expected key_signature, but got {token}"
                logger.warning(t)
                decoding_errors.append(t)
            else:
                key_signature = self._key_signature_token_to_model(token)
                tokens = tokens[1:]

        if self.encoding_settings.include_time_signature:
            token = tokens[0]
            if self._get_token_type(token) is not TokenType.TIME_SIGNATURE:
                t = f"Expected time_signature, but got {token}"
                logger.warning(t)
                decoding_errors.append(t)
            else:
                time_signature = self._time_signature_token_to_model(token)
                tokens = tokens[1:]

        current_offset: int = 0
        i: int = 0
        while i < len(tokens):
            current_token = tokens[i]
            token_type = self._get_token_type(current_token)
            match token_type:
                case TokenType.BAR:
                    offset_to_notes[current_offset].append(self._bar_model_token_to_model(current_token))
                case TokenType.PITCH:
                    if i + 1 < len(tokens):
                        next_token = tokens[i + 1]
                        if self._get_token_type(next_token) is TokenType.DURATION:
                            offset_to_notes[current_offset].append(
                                self._note_tokens_to_model(current_token, next_token)
                            )
                            i += 1
                case TokenType.REST:
                    if i + 1 < len(tokens):
                        next_token = tokens[i + 1]
                        if self._get_token_type(next_token) is TokenType.DURATION:
                            offset_to_notes[current_offset].append(self._rest_token_to_model(current_token, next_token))
                            i += 1
                case TokenType.TIME_SHIFT:
                    if i + 1 < len(tokens):
                        next_token = tokens[i + 1]
                        if self._get_token_type(next_token) is TokenType.DURATION:
                            current_offset += self._time_shift_token_to_model(current_token, next_token).duration
                            i += 1
                case _:
                    t = f"Got unexpected token {current_token} of type {token_type} at position {i}"
                    logger.warning(t)
                    decoding_errors.append(t)
            i += 1
        return offset_to_notes, clef, key_signature, time_signature, decoding_errors

    def get_tokenizer(self) -> None:
        raise NotImplementedError()
