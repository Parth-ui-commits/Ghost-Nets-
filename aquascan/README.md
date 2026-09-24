# AquaScan AI
Side-scan sonar image -> shadow-aware debris detection -> geotagged map + CSV report.

## Run locally
    pip install -r backend/requirements.txt
    uvicorn backend.app:app --reload      # from repo root, open http://localhost:8000
    python scripts/test_backend.py

## Real model
Train with `scripts/train.py`, put weights at `backend/weights/best.pt` (or set MODEL_PATH) and `pip install ultralytics`.
Without weights the app uses a bright-echo + shadow heuristic so the demo always works.

## Deploy
Backend: Hugging Face Space (Docker SDK) using the Dockerfile. Frontend: Vercel (root = repo, vercel.json points to /frontend);
add `<script>window.API_BASE="https://YOUR-SPACE.hf.space"</script>` before the main script in frontend/index.html.
