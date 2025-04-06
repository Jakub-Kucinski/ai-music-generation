from copy import deepcopy
from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ai_music_generation.core.pydantic_models.instrument_types import InstrumentTypes

PIANO_RANGE = (21, 108)


class EncodingSetting(BaseModel):
    include_bars: bool = True
    include_rests: bool = True
    include_clef: bool = True
    include_key_signature: bool = True
    include_time_signature: bool = True
    include_offset: bool = True
    include_offset_in_tuplets: bool = True
    joining_parts_strategy: Literal["Join measures", "Queue parallel measures"] = "Queue parallel measures"
    skip_measures_without_notes: bool = False
    notes_range: Tuple[int, int] = PIANO_RANGE
    shortest_note_duration: int = 16  # 1/n, shortest accepted note duration (Nth)
    longest_note_duration: int = 2  # n, longest accepted note duration (N whole notes)
    allow_triplet_quarterLength: bool = True
    allowed_instruments: list[InstrumentTypes] = Field(
        default_factory=lambda: deepcopy(
            [
                # InstrumentTypes.PIANO,
                # InstrumentTypes.CHROMATIC_PERCUSSION,
                # InstrumentTypes.ORGAN,
                # InstrumentTypes.GUITAR,
                # InstrumentTypes.BASS,
                # InstrumentTypes.STRINGS,
                # InstrumentTypes.ENSEMBLE,
                # InstrumentTypes.BRASS,
                # InstrumentTypes.REED,
                # InstrumentTypes.PIPE,
                # InstrumentTypes.SYNTH_LEAD,
                # InstrumentTypes.SYNTH_PAD,
                # InstrumentTypes.SYNTH_EFFECTS,
                # InstrumentTypes.ETHNIC,
                # InstrumentTypes.PERCUSSIVE,
                # InstrumentTypes.SOUND_EFFECTS,
            ]
        )
    )
    raise_duration_errors: bool = False

    model_config = ConfigDict(frozen=True)
