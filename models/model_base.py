"""
Abstract base class for ASR models.
This interface defines the functions that every ASR model
must implement. It allows different speech recognition
backends to be exchanged without changing the surrounding
streaming or UI code.
"""
from abc import ABC, abstractmethod
from typing import Optional
from core.types import AudioChunk


# --------------------------------------------------
# Common ASR model interface
# --------------------------------------------------

class ASRModel(ABC):

    @abstractmethod
    def start(self):
        """
        Initialize the model and start any required
        background processing.
        """
        ...

    @abstractmethod
    def stop(self):
        """
        Stop processing and release model resources.
        """
        ...

    @abstractmethod
    def accept_chunk(self, chunk: AudioChunk):
        """
        Receive a preprocessed audio chunk from
        the streaming pipeline.
        """
        ...

    @abstractmethod
    def poll(self) -> Optional[dict]:
        """
        Return newly available transcription events
        without blocking execution.
        """
        ...