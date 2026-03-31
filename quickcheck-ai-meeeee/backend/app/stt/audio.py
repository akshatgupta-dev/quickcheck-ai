from __future__ import annotations

import numpy as np

TARGET_SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
FRAME_MS = 20
FRAME_SAMPLES = TARGET_SAMPLE_RATE * FRAME_MS // 1000  # 320
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH_BYTES       # 640


def int16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def resample_pcm16(audio_bytes: bytes, source_sample_rate: int, target_sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    if not audio_bytes or source_sample_rate <= 0 or source_sample_rate == target_sample_rate:
        return audio_bytes

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if audio.size == 0:
        return b""

    duration = audio.shape[0] / float(source_sample_rate)
    target_length = max(1, int(round(duration * target_sample_rate)))

    old_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_length, endpoint=False, dtype=np.float32)
    resampled = np.interp(new_x, old_x, audio).astype(np.int16)
    return resampled.tobytes()


def split_frames(audio_bytes: bytes, frame_bytes: int = FRAME_BYTES) -> tuple[list[bytes], bytes]:
    if not audio_bytes:
        return [], b""
    frame_count = len(audio_bytes) // frame_bytes
    frames = [audio_bytes[i * frame_bytes:(i + 1) * frame_bytes] for i in range(frame_count)]
    remainder = audio_bytes[frame_count * frame_bytes:]
    return frames, remainder
