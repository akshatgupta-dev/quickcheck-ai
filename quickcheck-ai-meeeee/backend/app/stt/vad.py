from __future__ import annotations

import webrtcvad

from .audio import TARGET_SAMPLE_RATE


class VoiceActivityDetector:
    def __init__(self, aggressiveness: int = 2):
        self.vad = webrtcvad.Vad(max(0, min(3, aggressiveness)))

    def is_speech(self, frame: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> bool:
        if not frame:
            return False
        return self.vad.is_speech(frame, sample_rate)
