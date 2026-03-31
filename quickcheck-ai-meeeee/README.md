# Quick Check AI (Speech Recognition + Keypoint Listing)

MVP:
- Frontend: choose active project + live captions
- Backend: receives audio chunks + transcribes (Whisper)
- Storage: per-session transcripts per project

## Run (dev)
### Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install fastapi "uvicorn[standard]"
uvicorn app.main:app --reload --port 8000

### Frontend
cd frontend
npm install
npm run dev
