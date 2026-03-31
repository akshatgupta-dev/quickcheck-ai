from __future__ import annotations

import asyncio
import json
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .stt import FRAME_BYTES, TARGET_SAMPLE_RATE, VoiceActivityDetector, get_engine
from .stt.audio import resample_pcm16, split_frames

app = FastAPI(title="QuickCheck AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory transcript storage.
SESSIONS: Dict[str, Dict[str, list[dict[str, Any]]]] = {}

# Streaming constants tuned for low latency.
PRE_ROLL_MS = 240
END_SILENCE_MS = 520
MIN_UTTERANCE_MS = 180
PARTIAL_INTERVAL_MS = 240
MAX_PARTIAL_AUDIO_MS = 10_000
MAX_UTTERANCE_MS = 20_000
PARTIAL_MIN_CHARS = 2

PRE_ROLL_FRAMES = max(1, PRE_ROLL_MS // 20)
END_SILENCE_FRAMES = max(1, END_SILENCE_MS // 20)
MIN_UTTERANCE_FRAMES = max(1, MIN_UTTERANCE_MS // 20)
PARTIAL_INTERVAL_FRAMES = max(1, PARTIAL_INTERVAL_MS // 20)
MAX_PARTIAL_AUDIO_FRAMES = max(1, MAX_PARTIAL_AUDIO_MS // 20)
MAX_UTTERANCE_FRAMES = max(1, MAX_UTTERANCE_MS // 20)


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


@app.on_event("startup")
async def warm_model() -> None:
    await asyncio.to_thread(get_engine)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "sample_rate": TARGET_SAMPLE_RATE,
        "frame_bytes": FRAME_BYTES,
        "model": "faster-whisper",
    }


async def send_json_safe(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        pass


async def transcribe_bytes(audio_bytes: bytes, language: str, *, beam_size: int, vad_filter: bool) -> str:
    engine = get_engine()
    return await asyncio.to_thread(engine.transcribe, audio_bytes, language, beam_size, vad_filter)


async def run_partial_transcription(ws: WebSocket, state: SessionState, generation: int) -> None:
    try:
        audio = state.utterance_bytes
        if not audio:
            return

        frame_limit = MAX_PARTIAL_AUDIO_FRAMES * FRAME_BYTES
        if len(audio) > frame_limit:
            audio = audio[-frame_limit:]

        text = await transcribe_bytes(audio, state.language, beam_size=1, vad_filter=False)
        text = text.strip()

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
                "latency_mode": "streaming",
            },
        )
    except asyncio.CancelledError:
        return
    except Exception as e:
        print("partial transcription error:", e)


async def run_final_transcription(
    ws: WebSocket,
    state: SessionState,
    generation: int,
    audio: bytes,
    project_id: str,
    seq: int,
) -> None:
    try:
        text = await transcribe_bytes(audio, state.language, beam_size=3, vad_filter=False)
        text = text.strip()

        if state.closed or generation != state.transcribe_generation:
            return

        if not text:
            return

        print("RUN FINAL transcription, audio bytes:", len(audio))
        print("FINAL text:", repr(text))

        SESSIONS.setdefault(state.session_id, {}).setdefault(project_id, []).append(
            {"seq": seq, "text": text, "ts": int(time.time() * 1000)}
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
    except Exception as e:
        print("final transcription error:", e)


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

    print("FLUSH utterance frames:", len(state.utterance_frames), "force:", force)

    state.transcribe_generation += 1
    generation = state.transcribe_generation

    if state.active_partial_task and not state.active_partial_task.done():
        state.active_partial_task.cancel()

    if state.active_final_task and not state.active_final_task.done():
        state.active_final_task.cancel()

    state.reset_utterance()

    state.active_final_task = asyncio.create_task(
        run_final_transcription(ws, state, generation, audio, project_id, seq)
    )


async def process_frame(ws: WebSocket, state: SessionState, frame: bytes) -> None:
    vad = ws.state.vad
    is_speech = vad.is_speech(frame)
    print("is_speech:", is_speech, "speech_active:", state.speech_active, "frame len:", len(frame))

    state.pre_roll.append(frame)

    if not state.speech_active:
        if is_speech:
            state.speech_active = True
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

    if (
        utterance_frame_count >= MIN_UTTERANCE_FRAMES
        and state.frames_since_partial >= PARTIAL_INTERVAL_FRAMES
    ):
        state.frames_since_partial = 0
        state.transcribe_generation += 1
        generation = state.transcribe_generation

        if state.active_partial_task and not state.active_partial_task.done():
            state.active_partial_task.cancel()

        state.active_partial_task = asyncio.create_task(
            run_partial_transcription(ws, state, generation)
        )

    if state.silence_frames >= END_SILENCE_FRAMES or utterance_frame_count >= MAX_UTTERANCE_FRAMES:
        await flush_utterance(ws, state)


async def handle_control_message(ws: WebSocket, state: SessionState, message: dict[str, Any]) -> None:
    msg_type = message.get("type")

    if msg_type == "ping":
        await send_json_safe(ws, {"type": "pong"})
        return

    if msg_type == "config":
        state.project_id = message.get("project_id", state.project_id)
        state.language = message.get("language", state.language)
        state.source_sample_rate = int(message.get("sample_rate", state.source_sample_rate) or TARGET_SAMPLE_RATE)
        await send_json_safe(
            ws,
            {
                "type": "ready",
                "session_id": state.session_id,
                "sample_rate": TARGET_SAMPLE_RATE,
                "frame_ms": 20,
                "project_id": state.project_id,
                "language": state.language,
            },
        )
        return

    if msg_type == "set_context":
        new_project = message.get("project_id", state.project_id)
        new_language = message.get("language", state.language)

        if state.speech_active and new_project != state.project_id:
            await flush_utterance(ws, state, force=True)

        state.project_id = new_project
        state.language = new_language
        return

    if msg_type == "stop":
        await flush_utterance(ws, state, force=True)
        await send_json_safe(ws, {"type": "stopped", "session_id": state.session_id})
        return

    if msg_type == "get_project_transcript":
        session_id = message.get("session_id", state.session_id)
        project_id = message.get("project_id", state.project_id)
        entries = SESSIONS.get(session_id, {}).get(project_id, [])
        merged = " ".join(entry["text"] for entry in entries if entry.get("text")).strip()
        await send_json_safe(
            ws,
            {
                "type": "project_transcript",
                "session_id": session_id,
                "project_id": project_id,
                "text": merged,
            },
        )
        return

    await send_json_safe(ws, {"type": "error", "message": f"Unknown message type: {msg_type}"})


async def handle_audio_message(ws: WebSocket, state: SessionState, raw_message: bytes) -> None:
    print("raw audio packet len:", len(raw_message))

    if len(raw_message) < 4:
        print("packet too small")
        return

    metadata_len = struct.unpack("<I", raw_message[:4])[0]
    metadata_end = 4 + metadata_len
    if metadata_end > len(raw_message):
        print("invalid metadata_end:", metadata_end, "message len:", len(raw_message))
        return

    metadata = json.loads(raw_message[4:metadata_end].decode("utf-8"))
    chunk = raw_message[metadata_end:]

    print("metadata:", metadata)
    print("chunk len:", len(chunk))

    if not chunk:
        print("empty chunk")
        return

    state.project_id = metadata.get("project_id", state.project_id)
    state.language = metadata.get("language", state.language)
    state.source_sample_rate = int(metadata.get("sampleRate", state.source_sample_rate) or TARGET_SAMPLE_RATE)

    resampled = resample_pcm16(chunk, state.source_sample_rate, TARGET_SAMPLE_RATE)
    frames, remainder = split_frames(state.frame_remainder + resampled)
    state.frame_remainder = remainder

    print("resampled len:", len(resampled))
    print("frames count:", len(frames), "remainder len:", len(remainder))

    for frame in frames:
        await process_frame(ws, state, frame)


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()
    ws.state.vad = VoiceActivityDetector(aggressiveness=0)
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
        if state.speech_active and state.utterance_frames:
            try:
                await flush_utterance(ws, state, force=True)
            except Exception:
                pass

        state.closed = True

        if state.active_partial_task and not state.active_partial_task.done():
            state.active_partial_task.cancel()

        if state.active_final_task and not state.active_final_task.done():
            await asyncio.gather(state.active_final_task, return_exceptions=True)