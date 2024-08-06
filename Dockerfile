FROM python:3.12.2-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app


RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt
FROM python:3.12.2-slim-bookworm
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .
CMD [".venv/bin/python3", "-m", "backend.app"]


