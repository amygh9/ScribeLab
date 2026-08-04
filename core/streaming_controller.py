"""
Welcome to the Streaming Module!

This module connects to the Aria glasses audio stream, receives incoming
audio chunks, applies the preprocessing pipeline, and forwards the
processed audio to the currently active ASR model or consumer.

The design separates three responsibilities:

1. AriaAudioRouterObserver
   Receives raw audio callbacks from the Aria SDK and preprocesses them.

2. StreamingController
   Manages the lifecycle of the Aria streaming client
   (start, subscribe, stop, unsubscribe).

3. Sink Interface
   Any ASR model can register itself as a sink and receive processed
   AudioChunk objects through accept_chunk().
"""

import time
import threading
import aria.sdk as aria

from projectaria_tools.core.sensor_data import AudioDataRecord

from core.base_observer import BaseStreamingClientObserver
from core.audio_preprocessing import AudioPreprocessor, PreprocessConfig
from core.types import AudioChunk


# --------------------------------------------------
# Audio callback observer
# --------------------------------------------------

class AriaAudioRouterObserver(BaseStreamingClientObserver):
    def __init__(self, preprocessor: AudioPreprocessor, on_chunk_cb):
        self.pre = preprocessor
        self.on_chunk_cb = on_chunk_cb
        self.got_audio = False

    def on_audio_received(self, audio, record: AudioDataRecord):
        # Ignore empty audio packets
        if not audio.data:
            return

        # Print a message when the first audio arrives
        if not self.got_audio:
            print("[AUDIO] callback received")
            self.got_audio = True

        now_s = time.time()

        # Convert, resample and optionally denoise the audio
        audio_16k, rms, clip_frac = self.pre.process_aria_chunk(
            audio.data,
            now_s=now_s
        )

        # Warn if a noticeable amount of clipping occurs
        if clip_frac > self.pre.cfg.clipping_warn_frac:
            print(f"[WARN] clipping {clip_frac*100:.1f}% (reduce gain)")

        if audio_16k.size == 0:
            return

        # Package the processed audio into a standard chunk format
        chunk = AudioChunk(
            data=audio_16k,
            sample_rate=self.pre.cfg.target_sr,
            rms=rms,
            timestamp_s=now_s,
        )

        # Forward the chunk to the registered consumer
        self.on_chunk_cb(chunk)


# --------------------------------------------------
# Streaming client manager
# --------------------------------------------------

class StreamingController:
    """
    Controls the Aria streaming client and routes audio
    to the currently active sink.
    """

    def __init__(self, preprocess_cfg: PreprocessConfig | None = None):
        self.pre_cfg = preprocess_cfg or PreprocessConfig()
        self.pre = AudioPreprocessor(self.pre_cfg)

        self.client = None
        self.observer = None
        self.running = False

        # Stored mainly for logging and debugging
        self.interface = None

        self._lock = threading.Lock()

        # Target object receiving AudioChunk objects
        self._sink = None

    # --------------------------------------------------
    # Sink management
    # --------------------------------------------------

    def set_sink(self, sink):
        """
        Register a consumer that implements:
            accept_chunk(AudioChunk)
        """
        with self._lock:
            self._sink = sink

    def _on_chunk(self, chunk: AudioChunk):
        # Forward audio to the currently selected sink
        with self._lock:
            sink = self._sink

        if sink is not None:
            sink.accept_chunk(chunk)

    # --------------------------------------------------
    # Stream startup
    # --------------------------------------------------

    def start(self, interface: str = "usb"):
        """
        Start audio streaming from the Aria device.

        The interface parameter is currently only kept
        for compatibility with the CLI.
        """
        if self.running:
            return

        self.interface = interface

        print(f"[STREAM] starting (interface={interface})")

        aria.set_log_level(aria.Level.Info)

        # Create Aria streaming client
        client = aria.StreamingClient()

        # Configure audio subscription
        config = client.subscription_config

        config.subscriber_data_type = aria.StreamingDataType.Audio
        config.message_queue_size[aria.StreamingDataType.Audio] = 300

        # Enable secure communication
        opts = aria.StreamingSecurityOptions()
        opts.use_ephemeral_certs = True

        config.security_options = opts
        client.subscription_config = config

        # Register observer for incoming audio
        self.observer = AriaAudioRouterObserver(
            self.pre,
            self._on_chunk
        )

        client.set_streaming_client_observer(self.observer)

        self.client = client

        # Begin audio subscription
        self.client.subscribe()

        self.running = True

        print("[STREAM] started")

    # --------------------------------------------------
    # Stream shutdown
    # --------------------------------------------------

    def stop(self):
        """Stop streaming and release resources."""

        if not self.running:
            return

        try:
            self.client.unsubscribe()
        except Exception:
            pass

        self.client = None
        self.observer = None
        self.running = False

        print("[STREAM] stopped")