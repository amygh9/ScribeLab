"""
Welcome to the Preprocessing module for Aria microphone chunks!

This module takes raw interleaved int32 PCM audio from the Aria glasses,
converts it to mono float audio, resamples it from 48 kHz to 16 kHz,
applies gain, optionally reduces noise, and returns the processed audio
together with simple audio quality values such as RMS and clipping rate.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# Optional noise reduction library
try:
    import noisereduce as nr
except Exception:
    nr = None


# --------------------------------------------------
# Simple linear resampling
# --------------------------------------------------

def resample_linear(x: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return x.astype(np.float32, copy=False)

    ratio = tgt_sr / src_sr
    n_out = int(len(x) * ratio)

    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    # Map the original samples to the new sampling grid
    xp = np.linspace(0, 1, len(x), endpoint=False)
    xq = np.linspace(0, 1, n_out, endpoint=False)

    return np.interp(xq, xp, x).astype(np.float32)


# --------------------------------------------------
# Preprocessing configuration
# --------------------------------------------------

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


# --------------------------------------------------
# Audio preprocessing pipeline
# --------------------------------------------------

class AudioPreprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._last_rms_t = 0.0

    def aria_int32_to_mono_float(self, interleaved_int32: np.ndarray) -> np.ndarray:
        # Convert int32 PCM samples to float32 values in the range [-1, 1]
        audio_f32 = interleaved_int32.astype(np.float32) / 2147483648.0

        # Keep only complete multi-channel frames
        n = (len(audio_f32) // self.cfg.num_channels) * self.cfg.num_channels
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        # Reshape channels and average them into one mono signal
        mono = audio_f32[:n].reshape(-1, self.cfg.num_channels).mean(axis=1)

        return mono.astype(np.float32, copy=False)

    def denoise(self, audio_16k: np.ndarray) -> np.ndarray:
        # Skip denoising if it is disabled because this step is optional :)
        if not self.cfg.enable_denoise:
            return audio_16k
        if nr is None:
            return audio_16k

        # Use the beginning of the chunk as a simple noise profile as spectral noise reduction is usually done .
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

        # Convert the incoming samples to a NumPy int32 array
        pcm = np.asarray(aria_samples, dtype=np.int32)

        # Convert Aria multi-channel audio to mono 48 kHz float audio
        mono_48k = self.aria_int32_to_mono_float(pcm)

        if mono_48k.size == 0:
            return np.zeros(0, dtype=np.float32), 0.0, 0.0

        # Resample from Aria's 48 kHz to 16 kHz because the models usually expect 16 kHz. This is a simple linear resampling which is not the best quality but is very fast and has no external dependencies.
        # In a production application, we can use a better resampling method from a library like librosa or torchaudio but I think this does the job for my Bachelor's thesis demonstrator.
        audio_16k = resample_linear(mono_48k, self.cfg.aria_sr, self.cfg.target_sr)

        # Increase volume and check how much of the signal would clip
        audio_16k *= self.cfg.gain
        clip_frac = float(np.mean(np.abs(audio_16k) >= 1.0))

        # Limit values to the valid audio range
        audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32, copy=False)

        # Optionally apply noise reduction after resampling (when ticked in the UI)
        audio_16k = self.denoise(audio_16k)

        # RMS gives a simple estimate of the signal loudness
        rms = float(np.sqrt(np.mean(audio_16k ** 2))) if audio_16k.size else 0.0

        # Print audio status from time to time
        if log_rms and (now_s - self._last_rms_t > self.cfg.rms_log_period_s):
            print(f"[AUDIO] rms={rms:.5f} (clip={clip_frac*100:.2f}%)")
            self._last_rms_t = now_s

        return audio_16k, rms, clip_frac
    

    # Preprocessing Done!
    