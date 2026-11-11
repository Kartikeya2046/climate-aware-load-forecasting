# Energy Intelligence web app: one container serving the API and the built frontend.
#   docker build -t energy-intelligence .
#   docker run --rm -p 7860:7860 energy-intelligence      ->  http://localhost:7860

# ---- stage 1: build the React frontend
FROM node:24-slim AS web
WORKDIR /web
COPY dashboard/web/package.json dashboard/web/package-lock.json ./
RUN npm ci
COPY dashboard/web/ ./
RUN npm run build

# ---- stage 2: Python server with the models and data (.dockerignore limits experiments/ to what it reads)
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=2 PORT=7860
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY dashboard/requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY --chown=app experiments/ experiments/
COPY --chown=app dashboard/ dashboard/
COPY --from=web --chown=app /web/dist dashboard/web/dist
USER app
WORKDIR /app/dashboard
EXPOSE 7860
# One worker: predictions and uploads are held in this process's memory.
CMD ["sh", "-c", "exec python -m uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
