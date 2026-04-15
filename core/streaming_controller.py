import time
import threading
import aria.sdk as aria

from projectaria_tools.core.sensor_data import AudioDataRecord

from core.base_observer import BaseStreamingClientObserver
from core.audio_preprocessing import AudioPreprocessor, PreprocessConfig
from core.types import AudioChunk


class AriaAudioRouterObserver(BaseStreamingClientObserver):
    def __init__(self, preprocessor: AudioPreprocessor, on_chunk_cb):
        self.pre = preprocessor
        self.on_chunk_cb = on_chunk_cb
        self.got_audio = False

    def on_audio_received(self, audio, record: AudioDataRecord):
        if not audio.data:
            return
        if not self.got_audio:
            print("[AUDIO] callback received")
            self.got_audio = True

        now_s = time.time()
        audio_16k, rms, clip_frac = self.pre.process_aria_chunk(audio.data, now_s=now_s)

        if clip_frac > self.pre.cfg.clipping_warn_frac:
            print(f"[WARN] clipping {clip_frac*100:.1f}% (reduce gain)")

        if audio_16k.size == 0:
            return

        chunk = AudioChunk(
            data=audio_16k,
            sample_rate=self.pre.cfg.target_sr,
            rms=rms,
            timestamp_s=now_s,
        )
        self.on_chunk_cb(chunk)


class StreamingController:
    """Owns aria.StreamingClient lifecycle. Models attach via set_sink()."""
    def __init__(self, preprocess_cfg: PreprocessConfig | None = None):
        self.pre_cfg = preprocess_cfg or PreprocessConfig()
        self.pre = AudioPreprocessor(self.pre_cfg)

        self.client = None
        self.observer = None
        self.running = False

        self.interface = None  # store for logging/debug

        self._lock = threading.Lock()
        self._sink = None  # object with accept_chunk(AudioChunk)

    def set_sink(self, sink):
        with self._lock:
            self._sink = sink

    def _on_chunk(self, chunk: AudioChunk):
        with self._lock:
            sink = self._sink
        if sink is not None:
            sink.accept_chunk(chunk)

    def start(self, interface: str = "usb"):
        """Start Aria audio streaming. interface kept for CLI compatibility."""
        if self.running:
            return

        self.interface = interface
        print(f"[STREAM] starting (interface={interface})")

        aria.set_log_level(aria.Level.Info)

        client = aria.StreamingClient()
        config = client.subscription_config

        config.subscriber_data_type = aria.StreamingDataType.Audio
        config.message_queue_size[aria.StreamingDataType.Audio] = 300

        opts = aria.StreamingSecurityOptions()
        opts.use_ephemeral_certs = True
        config.security_options = opts
        client.subscription_config = config

        self.observer = AriaAudioRouterObserver(self.pre, self._on_chunk)
        client.set_streaming_client_observer(self.observer)

        self.client = client
        self.client.subscribe()
        self.running = True
        print("[STREAM] started")

    def stop(self):
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
