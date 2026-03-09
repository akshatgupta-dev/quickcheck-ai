# backend/app/main.py
import asyncio
import base64
import os
import subprocess
import tempfile
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

import whisper

app = FastAPI(title="QuickCheck AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (MVP). Later: SQLite.
SESSIONS: Dict[str, Dict[str, list]] = {}

_MODEL: Optional[Any] = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = whisper.load_model("base")  # tiny/base/small
    return _MODEL


def _suffix_from_mime(mime: str) -> str:
    m = (mime or "").lower()
    if "ogg" in m:
        return ".ogg"
    if "webm" in m:
        return ".webm"
    return ".bin"


def bytes_to_wav_path(data: bytes, mime: str) -> str:
    """
    Convert audio bytes (webm/opus or ogg/opus) to wav (16kHz mono).
    Returns path to wav file. Caller should delete wav file afterward.
    """
    in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix_from_mime(mime))
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        in_tmp.write(data)
        in_tmp.flush()
        in_tmp.close()
        out_tmp.close()

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", in_tmp.name,
            "-ar", "16000",
            "-ac", "1",
            out_tmp.name
        ]
        subprocess.run(cmd, check=True)
        return out_tmp.name
    finally:
        # always delete input chunk container
        try:
            os.remove(in_tmp.name)
        except Exception:
            pass


def transcribe_wav(wav_path: str, language: str) -> str:
    model = get_model()
    lang_arg = None if language == "auto" else language  # "fi", "en", or None
    result = model.transcribe(
        wav_path,
        language=lang_arg,
        task="transcribe",
        fp16=False,
        temperature=0.0,
    )
    return (result.get("text") or "").strip()


@app.get("/health")
def health():
    return {"ok": True}


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if msg_type == "audio_chunk":
                session_id = msg.get("session_id", "default-session")
                project_id = msg.get("project_id", "unknown-project")
                language = msg.get("language", "auto")
                seq = msg.get("seq", 0)
                mime = msg.get("mime", "audio/webm")

                data_b64 = msg.get("data_b64", "")
                if not data_b64:
                    await ws.send_json({"type": "error", "message": "Missing data_b64"})
                    continue

                audio_bytes = base64.b64decode(data_b64)

                try:
                    def work():
                        wav_path = bytes_to_wav_path(audio_bytes, mime)
                        try:
                            return transcribe_wav(wav_path, language)
                        finally:
                            try:
                                os.remove(wav_path)
                            except Exception:
                                pass

                    text = await asyncio.to_thread(work)

                except subprocess.CalledProcessError:
                    await ws.send_json({
                        "type": "error",
                        "project_id": project_id,
                        "seq": seq,
                        "message": f"ffmpeg failed decoding chunk (mime={mime})."
                    })
                    continue

                except Exception as e:
                    await ws.send_json({
                        "type": "error",
                        "project_id": project_id,
                        "seq": seq,
                        "message": f"transcription failed: {type(e).__name__}"
                    })
                    continue

                # Store per project
                SESSIONS.setdefault(session_id, {}).setdefault(project_id, []).append({
                    "seq": seq,
                    "text": text
                })

                # Stream back to UI
                await ws.send_json({
                    "type": "partial",
                    "session_id": session_id,
                    "project_id": project_id,
                    "seq": seq,
                    "text": text,
                    "is_final": False
                })
                continue

            if msg_type == "get_project_transcript":
                session_id = msg.get("session_id", "default-session")
                project_id = msg.get("project_id", "unknown-project")
                entries = SESSIONS.get(session_id, {}).get(project_id, [])
                merged = " ".join([e["text"] for e in entries if e["text"]]).strip()
                await ws.send_json({
                    "type": "project_transcript",
                    "session_id": session_id,
                    "project_id": project_id,
                    "text": merged
                })
                continue

            await ws.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    except Exception:
        pass