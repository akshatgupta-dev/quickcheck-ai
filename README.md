# Quick Check AI

QuickCheck AI records microphone audio continuously, transcribes speech per project, and turns each project transcript into a bullet summary.

Core flow:
- Frontend: choose or create a project, stream mic audio, and watch live captions
- Backend: receives PCM frames, detects speech turns, and transcribes with Whisper
- Storage: per-session transcripts and summaries are grouped by project
- Summaries: request a project summary or a whole-session summary at any point

## Run (dev)
### Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

### Frontend
cd frontend
npm install
npm run dev

## Notes
- The browser uses an AudioWorklet to stream PCM audio, so there is no ffmpeg dependency in the hot path.
- Summaries are generated from the stored project transcript, which keeps them fast and deterministic.