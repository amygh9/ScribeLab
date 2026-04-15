from abc import ABC, abstractmethod
from typing import Optional
from core.types import AudioChunk


class ASRModel(ABC):
    @abstractmethod
    def start(self):
        """Allocate model resources, start worker thread if needed."""
        ...

    @abstractmethod
    def stop(self):
        """Stop worker thread, free resources."""
        ...

    @abstractmethod
    def accept_chunk(self, chunk: AudioChunk):
        """Receive preprocessed audio chunk (float32 mono, 16k)."""
        ...

    @abstractmethod
    def poll(self) -> Optional[dict]:
        """Non-blocking: returns events like {"type":"partial","text":...}."""
        ...
