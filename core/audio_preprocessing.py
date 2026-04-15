import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import noisereduce as nr
except Exception:
    nr = None


def resample_linear(x: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return x.astype(np.float32, copy=False)
    ratio = tgt_sr / src_sr
    n_out = int(len(x) * ratio)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0, 1, len(x), endpoint=False)
    xq = np.linspace(0, 1, n_out, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


@dataclass
class PreprocessConfig:
    aria_sr: int = 48000
    target_sr: int = 16000
    num_channels: int = 7
    gain: float = 8.0

    enable_denoise: bool = False
    noise_profile_seconds: float = 0.5
    prop_decrease: float = 0.8

    rms_log_period_s: float = 1.0
    clipping_warn_frac: float = 0.01


class AudioPreprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._last_rms_t = 0.0

    def aria_int32_to_mono_float(self, interleaved_int32: np.ndarray) -> np.ndarray:
        # int32 PCM -> float32 [-1,1]
        audio_f32 = interleaved_int32.astype(np.float32) / 2147483648.0

        n = (len(audio_f32) // self.cfg.num_channels) * self.cfg.num_channels
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        mono = audio_f32[:n].reshape(-1, self.cfg.num_channels).mean(axis=1)
        return mono.astype(np.float32, copy=False)

    def denoise(self, audio_16k: np.ndarray) -> np.ndarray:
        if not self.cfg.enable_denoise:
            return audio_16k
        if nr is None:
            # noisereduce not installed
            return audio_16k

        n_noise = int(self.cfg.noise_profile_seconds * self.cfg.target_sr)
        n_noise = max(1, min(len(audio_16k), n_noise))
        noise_clip = audio_16k[:n_noise]

        reduced = nr.reduce_noise(
            y=audio_16k,
            y_noise=noise_clip,
            sr=self.cfg.target_sr,
            prop_decrease=self.cfg.prop_decrease,
            stationary=False,
        )
        return reduced.astype(np.float32, copy=False)

    def process_aria_chunk(
        self,
        aria_samples: "list[int] | np.ndarray",
        now_s: Optional[float] = None,
        log_rms: bool = True,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Returns: (audio_16k_float32_mono, rms, clip_frac)
        """
        if now_s is None:
            now_s = time.time()

        pcm = np.asarray(aria_samples, dtype=np.int32)
        mono_48k = self.aria_int32_to_mono_float(pcm)
        if mono_48k.size == 0:
            return np.zeros(0, dtype=np.float32), 0.0, 0.0

        audio_16k = resample_linear(mono_48k, self.cfg.aria_sr, self.cfg.target_sr)

        audio_16k *= self.cfg.gain
        clip_frac = float(np.mean(np.abs(audio_16k) >= 1.0))

        audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32, copy=False)

        # optional denoise at 16k
        audio_16k = self.denoise(audio_16k)

        rms = float(np.sqrt(np.mean(audio_16k ** 2))) if audio_16k.size else 0.0

        if log_rms and (now_s - self._last_rms_t > self.cfg.rms_log_period_s):
            print(f"[AUDIO] rms={rms:.5f} (clip={clip_frac*100:.2f}%)")
            self._last_rms_t = now_s

        return audio_16k, rms, clip_frac
