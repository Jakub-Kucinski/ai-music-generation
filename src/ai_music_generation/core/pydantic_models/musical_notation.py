from pydantic import BaseModel


class NoteModel(BaseModel):
    pitch: int
    duration: int


class RestModel(BaseModel):
    duration: int


class TimeShiftModel(BaseModel):
    duration: int


class TimeSignatureModel(BaseModel):
    numerator: int
    denominator: int


class ClefModel(BaseModel):
    sign: str
    line: int
    octaveChange: int
    is_percussion: bool = False


class KeySignatureModel(BaseModel):
    sharps: int = 0


class BarModel(BaseModel):
    pass
