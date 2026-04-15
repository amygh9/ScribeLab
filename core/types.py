from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioChunk:
    """Unified audio chunk passed to models/diarizer.
    data: float32 mono waveform in [-1, 1]
    sample_rate: Hz (usually 16000 after preprocessing)
    rms: float RMS of chunk
    timestamp_s: optional wall timestamp
    """
    data: "object"  # keep as object to avoid hard numpy dependency in typing
    sample_rate: int
    rms: float
    timestamp_s: Optional[float] = None
