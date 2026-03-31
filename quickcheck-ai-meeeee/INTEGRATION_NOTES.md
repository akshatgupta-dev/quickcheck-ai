# QuickCheck AI + low-latency streaming STT

## Updated folder structure

```text
backend/
  app/
    main.py
    stt/
      __init__.py
      audio.py
      engine.py
      vad.py
  requirements.txt
frontend/
  public/
    pcm-processor.js
  src/
    App.tsx
    App.css
```

## What changed

- Removed `MediaRecorder -> base64 -> ffmpeg -> whisper` from the hot path.
- Added `AudioWorklet -> PCM16 -> WebSocket binary frames` from browser to backend.
- Added `webrtcvad` endpointing and `faster-whisper` transcription.
- Added partial streaming updates during speech and final updates on end-of-speech.

## Install

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Notes

- Default model is `base`. You can change it with environment variables:
  - `WHISPER_MODEL_SIZE=tiny|base|small|medium|large-v3`
  - `WHISPER_DEVICE=cpu|cuda`
  - `WHISPER_COMPUTE_TYPE=int8|float16|float32`
- For production GPU boxes, a common starting point is:
  - `WHISPER_MODEL_SIZE=small`
  - `WHISPER_DEVICE=cuda`
  - `WHISPER_COMPUTE_TYPE=float16`
