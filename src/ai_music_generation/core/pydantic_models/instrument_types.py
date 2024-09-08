from enum import Enum
from typing import Set


class InstrumentTypes(Enum):
    PIANO: Set[int] = set(range(0, 8))
    CHROMATIC_PERCUSSION: Set[int] = set(range(8, 16))
    ORGAN: Set[int] = set(range(17, 24))
    GUITAR: Set[int] = set(range(24, 32))
    BASS: Set[int] = set(range(32, 40))
    STRINGS: Set[int] = set(range(40, 48))
    ENSEMBLE: Set[int] = set(range(48, 56))
    BRASS: Set[int] = set(range(56, 64))
    REED: Set[int] = set(range(64, 72))
    PIPE: Set[int] = set(range(72, 80))
    SYNTH_LEAD: Set[int] = set(range(80, 88))
    SYNTH_PAD: Set[int] = set(range(88, 96))
    SYNTH_EFFECTS: Set[int] = set(range(96, 104))
    ETHNIC: Set[int] = set(range(104, 112))
    PERCUSSIVE: Set[int] = set(range(112, 120))
    SOUND_EFFECTS: Set[int] = set(range(120, 128))
