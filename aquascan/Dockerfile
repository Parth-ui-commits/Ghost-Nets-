FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY . .
ENV PORT=7860
CMD ["sh","-c","cd /app && uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
