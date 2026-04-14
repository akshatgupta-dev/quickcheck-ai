from __future__ import annotations

import asyncio
import json
import os
import re
import struct
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import whisper

from .stt.audio import FRAME_BYTES, TARGET_SAMPLE_RATE, int16_bytes_to_float32, resample_pcm16, split_frames
from .stt.vad import VoiceActivityDetector

app = FastAPI(title="QuickCheck AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, Dict[str, list[dict[str, Any]]]] = {}

MODEL_NAME = os.getenv("WHISPER_MODEL_SIZE", "base")
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "1"))
SUMMARY_BULLETS = max(3, int(os.getenv("SUMMARY_BULLETS", "5")))

FRAME_MS = 20
PRE_ROLL_MS = 240
END_SILENCE_MS = 520
MIN_UTTERANCE_MS = 180
PARTIAL_INTERVAL_MS = 240
MAX_PARTIAL_AUDIO_MS = 10_000
MAX_UTTERANCE_MS = 20_000
PARTIAL_MIN_CHARS = 2

PRE_ROLL_FRAMES = max(1, PRE_ROLL_MS // FRAME_MS)
END_SILENCE_FRAMES = max(1, END_SILENCE_MS // FRAME_MS)
MIN_UTTERANCE_FRAMES = max(1, MIN_UTTERANCE_MS // FRAME_MS)
PARTIAL_INTERVAL_FRAMES = max(1, PARTIAL_INTERVAL_MS // FRAME_MS)
MAX_PARTIAL_AUDIO_FRAMES = max(1, MAX_PARTIAL_AUDIO_MS // FRAME_MS)
MAX_UTTERANCE_FRAMES = max(1, MAX_UTTERANCE_MS // FRAME_MS)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "do",
    "for",
    "from",
    "have",
    "has",
    "he",
    "her",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "no",
    "of",
    "on",
    "or",
    "our",
    "out",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
    "ja",
    "ei",
    "että",
    "kun",
    "niin",
    "se",
    "ne",
    "on",
    "oli",
    "olen",
    "ole",
    "olla",
    "kuin",
    "mutta",
    "jos",
    "sinä",
    "hän",
    "me",
    "te",
    "he",
    "mikä",
    "mitä",
    "missä",
    "milloin",
    "miksi",
    "tämä",
    "tuo",
    "nämä",
    "nuo",
}

SUMMARY_HINTS = (
    "next step",
    "action",
    "action item",
    "decision",
    "decide",
    "need to",
    "should",
    "plan",
    "important",
    "follow up",
    "summary",
)

_MODEL: Optional[Any] = None
_MODEL_LOCK = threading.Lock()


def get_model() -> Any:
    global _MODEL
    if _MODEL is None:
        _MODEL = whisper.load_model(MODEL_NAME)
    return _MODEL


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9À-ÿ']+", text.lower())


def split_sentences(text: str) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [normalize_whitespace(candidate) for candidate in candidates if normalize_whitespace(candidate)]


def merge_transcript_entries(entries: list[dict[str, Any]]) -> str:
    return normalize_whitespace(" ".join(entry.get("text", "") for entry in entries if entry.get("text")))


def build_summary_bullets(entries: list[dict[str, Any]]) -> list[str]:
    transcript = merge_transcript_entries(entries)
    if not transcript:
        return ["No transcript captured yet."]

    sentences: list[str] = []
    seen_sentences: set[str] = set()
    for chunk in split_sentences(transcript) or [transcript]:
        key = chunk.lower()
        if len(chunk) < 10 or key in seen_sentences:
            continue
        seen_sentences.add(key)
        sentences.append(chunk)

    if not sentences:
        return ["No clear summary could be generated yet."]

    frequencies: Counter[str] = Counter()
    for sentence in sentences:
        for token in tokenize(sentence):
            if len(token) <= 2 or token in STOPWORDS or token.isdigit():
                continue
            frequencies[token] += 1

    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        tokens = [token for token in tokenize(sentence) if len(token) > 2 and token not in STOPWORDS and not token.isdigit()]
        if not tokens:
            continue

        score = sum(frequencies[token] for token in tokens) / (len(tokens) ** 0.5)
        lowered = sentence.lower()
        if any(hint in lowered for hint in SUMMARY_HINTS):
            score += 1.5
        if any(character.isdigit() for character in sentence):
            score += 0.25

        scored.append((score, index, sentence))

    if not scored:
        scored = [(1.0, index, sentence) for index, sentence in enumerate(sentences)]

    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:SUMMARY_BULLETS]
    selected.sort(key=lambda item: item[1])

    bullets: list[str] = []
    for _, _, sentence in selected:
        cleaned = sentence.rstrip(" .")
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        bullets.append(cleaned)

    return bullets or ["No clear summary could be generated yet."]


def build_project_summary(session_id: str, project_id: str) -> dict[str, Any]:
    entries = SESSIONS.get(session_id, {}).get(project_id, [])
    transcript = merge_transcript_entries(entries)
    return {
        "session_id": session_id,
        "project_id": project_id,
        "bullets": build_summary_bullets(entries),
        "transcript": transcript,
        "line_count": len(entries),
        "word_count": len(tokenize(transcript)),
        "generated_at": int(time.time() * 1000),
    }


def build_session_summary(session_id: str) -> dict[str, Any]:
    session = SESSIONS.get(session_id, {})
    return {
        "session_id": session_id,
        "projects": [build_project_summary(session_id, project_id) for project_id in session.keys()],
    }


def transcribe_pcm16(audio_bytes: bytes, language: str, *, beam_size: int) -> str:
    samples = int16_bytes_to_float32(audio_bytes)
    if samples.size == 0:
        return ""

    lang_arg = None if language == "auto" else language

    def work() -> str:
        model = get_model()
        with _MODEL_LOCK:
            result = model.transcribe(
                samples,
                language=lang_arg,
                task="transcribe",
                fp16=False,
                temperature=0.0,
                beam_size=beam_size,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
        return normalize_whitespace(result.get("text") or "")

    return work()


@dataclass
class SessionState:
    session_id: str
    project_id: str = "unknown-project"
    language: str = "auto"
    source_sample_rate: int = TARGET_SAMPLE_RATE
    seq: int = 0
    speech_active: bool = False
    utterance_frames: list[bytes] = field(default_factory=list)
    pre_roll: Deque[bytes] = field(default_factory=lambda: deque(maxlen=PRE_ROLL_FRAMES))
    silence_frames: int = 0
    frame_remainder: bytes = b""
    frames_since_partial: int = 0
    last_partial_text: str = ""
    transcribe_generation: int = 0
    active_partial_task: Optional[asyncio.Task] = None
    active_final_task: Optional[asyncio.Task] = None
    closed: bool = False

    def reset_utterance(self) -> None:
        self.speech_active = False
        self.utterance_frames.clear()
        self.silence_frames = 0
        self.frames_since_partial = 0
        self.last_partial_text = ""

    @property
    def utterance_bytes(self) -> bytes:
        return b"".join(self.utterance_frames)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "sample_rate": TARGET_SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "model": MODEL_NAME,
    }


async def send_json_safe(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        pass


async def transcribe_audio(audio_bytes: bytes, language: str, *, beam_size: int) -> str:
    return await asyncio.to_thread(transcribe_pcm16, audio_bytes, language, beam_size=beam_size)


async def run_partial_transcription(ws: WebSocket, state: SessionState, generation: int) -> None:
    try:
        audio = state.utterance_bytes
        if not audio:
            return

        limit_bytes = MAX_PARTIAL_AUDIO_FRAMES * FRAME_BYTES
        if len(audio) > limit_bytes:
            audio = audio[-limit_bytes:]

        text = (await transcribe_audio(audio, state.language, beam_size=1)).strip()
        if state.closed or generation != state.transcribe_generation:
            return
        if len(text) < PARTIAL_MIN_CHARS or text == state.last_partial_text:
            return

        state.last_partial_text = text
        await send_json_safe(
            ws,
            {
                "type": "partial",
                "session_id": state.session_id,
                "project_id": state.project_id,
                "seq": state.seq,
                "text": text,
                "is_final": False,
            },
        )
    except asyncio.CancelledError:
        return
    except Exception as exc:
        await send_json_safe(
            ws,
            {
                "type": "error",
                "session_id": state.session_id,
                "project_id": state.project_id,
                "message": f"partial transcription failed: {type(exc).__name__}",
            },
        )


async def run_final_transcription(ws: WebSocket, state: SessionState, generation: int, audio: bytes, project_id: str, seq: int) -> None:
    try:
        text = (await transcribe_audio(audio, state.language, beam_size=3)).strip()
        if state.closed or generation != state.transcribe_generation:
            return
        if not text:
            return

        SESSIONS.setdefault(state.session_id, {}).setdefault(project_id, []).append(
            {
                "seq": seq,
                "text": text,
                "ts": int(time.time() * 1000),
            }
        )

        await send_json_safe(
            ws,
            {
                "type": "final",
                "session_id": state.session_id,
                "project_id": project_id,
                "seq": seq,
                "text": text,
                "is_final": True,
            },
        )
    except asyncio.CancelledError:
        return
    except Exception as exc:
        await send_json_safe(
            ws,
            {
                "type": "error",
                "session_id": state.session_id,
                "project_id": project_id,
                "message": f"final transcription failed: {type(exc).__name__}",
            },
        )


async def flush_utterance(ws: WebSocket, state: SessionState, *, force: bool = False) -> None:
    if not state.utterance_frames:
        state.reset_utterance()
        return

    if not force and len(state.utterance_frames) < MIN_UTTERANCE_FRAMES:
        state.reset_utterance()
        return

    audio = state.utterance_bytes
    project_id = state.project_id
    seq = state.seq

    state.transcribe_generation += 1
    generation = state.transcribe_generation

    tasks = [task for task in (state.active_partial_task, state.active_final_task) if task and not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    state.active_partial_task = None
    state.active_final_task = None
    state.reset_utterance()

    if force:
        await run_final_transcription(ws, state, generation, audio, project_id, seq)
        return

    state.active_final_task = asyncio.create_task(run_final_transcription(ws, state, generation, audio, project_id, seq))


async def process_frame(ws: WebSocket, state: SessionState, frame: bytes) -> None:
    vad: VoiceActivityDetector = ws.state.vad
    is_speech = vad.is_speech(frame)

    state.pre_roll.append(frame)

    if not state.speech_active:
        if is_speech:
            state.speech_active = True
            state.seq += 1
            state.utterance_frames = list(state.pre_roll)
            state.silence_frames = 0
            state.frames_since_partial = 0
            state.last_partial_text = ""
        return

    state.utterance_frames.append(frame)
    utterance_frame_count = len(state.utterance_frames)

    if is_speech:
        state.silence_frames = 0
    else:
        state.silence_frames += 1

    state.frames_since_partial += 1

    if utterance_frame_count >= MIN_UTTERANCE_FRAMES and state.frames_since_partial >= PARTIAL_INTERVAL_FRAMES:
        state.frames_since_partial = 0
        state.transcribe_generation += 1
        generation = state.transcribe_generation

        if state.active_partial_task and not state.active_partial_task.done():
            state.active_partial_task.cancel()

        state.active_partial_task = asyncio.create_task(run_partial_transcription(ws, state, generation))

    if state.silence_frames >= END_SILENCE_FRAMES or utterance_frame_count >= MAX_UTTERANCE_FRAMES:
        await flush_utterance(ws, state)


async def handle_control_message(ws: WebSocket, state: SessionState, message: dict[str, Any]) -> None:
    msg_type = message.get("type")

    if msg_type == "ping":
        await send_json_safe(ws, {"type": "pong"})
        return

    if msg_type == "config":
        incoming_session_id = message.get("session_id")
        if incoming_session_id:
            state.session_id = incoming_session_id

        state.project_id = message.get("project_id", state.project_id)
        state.language = message.get("language", state.language)
        state.source_sample_rate = int(message.get("sample_rate", state.source_sample_rate) or TARGET_SAMPLE_RATE)

        await send_json_safe(
            ws,
            {
                "type": "ready",
                "session_id": state.session_id,
                "project_id": state.project_id,
                "language": state.language,
                "sample_rate": TARGET_SAMPLE_RATE,
                "frame_ms": FRAME_MS,
            },
        )
        return

    if msg_type == "set_context":
        incoming_session_id = message.get("session_id")
        if incoming_session_id:
            state.session_id = incoming_session_id

        new_project = message.get("project_id", state.project_id)
        new_language = message.get("language", state.language)

        if state.speech_active and new_project != state.project_id:
            await flush_utterance(ws, state, force=True)

        state.project_id = new_project
        state.language = new_language
        return

    if msg_type == "stop":
        incoming_session_id = message.get("session_id")
        if incoming_session_id:
            state.session_id = incoming_session_id

        await flush_utterance(ws, state, force=True)
        await send_json_safe(ws, {"type": "stopped", "session_id": state.session_id})
        return

    if msg_type == "get_project_transcript":
        session_id = message.get("session_id", state.session_id)
        project_id = message.get("project_id", state.project_id)
        entries = SESSIONS.get(session_id, {}).get(project_id, [])
        await send_json_safe(
            ws,
            {
                "type": "project_transcript",
                "session_id": session_id,
                "project_id": project_id,
                "text": merge_transcript_entries(entries),
                "line_count": len(entries),
            },
        )
        return

    if msg_type == "get_project_summary":
        session_id = message.get("session_id", state.session_id)
        project_id = message.get("project_id", state.project_id)
        summary = build_project_summary(session_id, project_id)
        await send_json_safe(ws, {"type": "project_summary", **summary})
        return

    if msg_type == "get_session_summary":
        session_id = message.get("session_id", state.session_id)
        await send_json_safe(ws, {"type": "session_summary", **build_session_summary(session_id)})
        return

    await send_json_safe(ws, {"type": "error", "message": f"Unknown message type: {msg_type}"})


async def handle_audio_message(ws: WebSocket, state: SessionState, raw_message: bytes) -> None:
    if len(raw_message) < 4:
        return

    try:
        metadata_len = struct.unpack("<I", raw_message[:4])[0]
        metadata_end = 4 + metadata_len
        if metadata_end > len(raw_message):
            return

        metadata = json.loads(raw_message[4:metadata_end].decode("utf-8"))
        chunk = raw_message[metadata_end:]
    except Exception as exc:
        await send_json_safe(ws, {"type": "error", "message": f"Invalid audio packet: {type(exc).__name__}"})
        return

    if not chunk:
        return

    incoming_session_id = metadata.get("session_id")
    if incoming_session_id:
        state.session_id = incoming_session_id

    state.project_id = metadata.get("project_id", state.project_id)
    state.language = metadata.get("language", state.language)
    state.source_sample_rate = int(metadata.get("sampleRate", state.source_sample_rate) or TARGET_SAMPLE_RATE)

    resampled = resample_pcm16(chunk, state.source_sample_rate, TARGET_SAMPLE_RATE)
    frames, remainder = split_frames(state.frame_remainder + resampled)
    state.frame_remainder = remainder

    for frame in frames:
        await process_frame(ws, state, frame)


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()
    ws.state.vad = VoiceActivityDetector(aggressiveness=VAD_AGGRESSIVENESS)
    state = SessionState(session_id=f"session-{int(time.time() * 1000)}")

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            if message.get("text"):
                payload = json.loads(message["text"])
                incoming_session_id = payload.get("session_id")
                if incoming_session_id:
                    state.session_id = incoming_session_id
                state.seq = int(payload.get("seq", state.seq))
                await handle_control_message(ws, state, payload)
                continue

            if message.get("bytes"):
                await handle_audio_message(ws, state, message["bytes"])
                continue

    except WebSocketDisconnect:
        pass
    finally:
        try:
            if state.speech_active and state.utterance_frames:
                await flush_utterance(ws, state, force=True)
        except Exception:
            pass

        state.closed = True

        if state.active_partial_task and not state.active_partial_task.done():
            state.active_partial_task.cancel()

        if state.active_final_task and not state.active_final_task.done():
            state.active_final_task.cancel()
            await asyncio.gather(state.active_final_task, return_exceptions=True)