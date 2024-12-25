from copy import deepcopy
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field

from ai_music_generation.core.pydantic_models.instrument_types import InstrumentTypes

PIANO_RANGE = (21, 108)


class EncodingSetting(BaseModel):
    include_bars: bool = True
    include_rests: bool = True
    include_clef: bool = True
    include_key_signature: bool = True
    include_time_signature: bool = True
    join_parts: bool = False
    notes_range: Tuple[int, int] = PIANO_RANGE
    shortest_note_duration: int = 16  # 1/n, shortest accepted note duration (Nth)
    longest_note_duration: int = 4  # n, longest accepted note duration (N whole notes)
    allowed_instruments: list[InstrumentTypes] = Field(
        default_factory=lambda: deepcopy(
            [
                InstrumentTypes.PIANO,
                InstrumentTypes.CHROMATIC_PERCUSSION,
                InstrumentTypes.ORGAN,
            ]
        )
    )

    model_config = ConfigDict(frozen=True)
