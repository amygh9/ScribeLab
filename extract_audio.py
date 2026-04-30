#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf

from projectaria_tools.core import data_provider


def extract_audio(vrs_path: Path, out_wav: Path, channel: int = 0):
    provider = data_provider.create_vrs_data_provider(str(vrs_path))

    stream_id = provider.get_stream_id_from_label("mic")
    n_records = provider.get_num_data(stream_id)

    print(f"[INFO] mic stream: {stream_id}")
    print(f"[INFO] records: {n_records}")

    if n_records <= 0:
        raise RuntimeError("Mic stream exists, but contains no audio records.")

    chunks = []

    for index in range(n_records):
        audio_data = provider.get_audio_data_by_index(stream_id, index)

        audio_block = np.asarray(audio_data[0].data, dtype=np.float32)

        # Aria mic data is interleaved:
        # 7 channels: ch0, ch1, ..., ch6, ch0, ch1, ...
        timestamps = audio_data[1].capture_timestamps_ns
        n_channels = len(audio_block) // len(timestamps)

        if n_channels <= 0:
            raise RuntimeError(f"Could not infer channel count at record {index}")

        if channel >= n_channels:
            raise ValueError(
                f"Requested channel {channel}, but recording has only {n_channels} channels"
            )

        mono = audio_block[channel::n_channels]
        chunks.append(mono)

        if index == 0:
            print(f"[INFO] inferred channels: {n_channels}")
            print(f"[INFO] first block raw samples: {len(audio_block)}")
            print(f"[INFO] first block mono samples: {len(mono)}")

    audio = np.concatenate(chunks).astype(np.float32)

    # Normalize only if values are integer-like / too large for WAV float
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"[INFO] max abs before norm: {max_abs}")

    if max_abs > 1.0:
        audio = audio / max_abs

    # Aria mic stream is normally 48 kHz.
    # If metadata does not expose sample rate, infer from timestamps.
    try:
        ts0 = provider.get_audio_data_by_index(stream_id, 0)[1].capture_timestamps_ns
        duration_first_block_s = (ts0[-1] - ts0[0]) * 1e-9
        sample_rate = int(round((len(ts0) - 1) / duration_first_block_s))
    except Exception:
        sample_rate = 48000

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, sample_rate)

    print(f"[DONE] saved: {out_wav}")
    print(f"[INFO] sample rate: {sample_rate}")
    print(f"[INFO] duration: {len(audio) / sample_rate:.2f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()

    extract_audio(args.vrs, args.out, channel=args.channel)


if __name__ == "__main__":
    main()