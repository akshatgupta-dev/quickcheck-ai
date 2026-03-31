from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Optional

from faster_whisper import WhisperModel

from .audio import int16_bytes_to_float32

MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
MODEL_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
MODEL_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")


class FasterWhisperEngine:
    def __init__(self, model_size: str = MODEL_SIZE, device: str = MODEL_DEVICE, compute_type: str = MODEL_COMPUTE_TYPE):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._lock = threading.Lock()

    def transcribe(self, pcm16_audio: bytes, language: str = "auto", beam_size: int = 1, vad_filter: bool = False) -> str:
        if not pcm16_audio:
            return ""

        samples = int16_bytes_to_float32(pcm16_audio)
        lang: Optional[str] = None if language == "auto" else language

        with self._lock:
            segments, _ = self.model.transcribe(
                samples,
                language=lang,
                task="transcribe",
                beam_size=beam_size,
                vad_filter=vad_filter,
                temperature=0.0,
                condition_on_previous_text=True,
                word_timestamps=False,
            )
            text = " ".join((segment.text or "").strip() for segment in segments).strip()
        return " ".join(text.split())


@lru_cache(maxsize=1)
def get_engine() -> FasterWhisperEngine:
    return FasterWhisperEngine()
