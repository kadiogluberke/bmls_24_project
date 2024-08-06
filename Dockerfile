# Building the image, with installed dependencies
FROM python:3.11.9-slim-bullseye AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
    
WORKDIR /app

RUN python -m venv .venv

COPY requirements-app.txt ./

RUN .venv/bin/pip install -r requirements-app.txt

# Running the application
FROM python:3.11.9-slim-bullseye

WORKDIR /app

COPY --from=builder /app/.venv .venv/

COPY . .

CMD [".venv/bin/python3", "-m", "backend.app"]


