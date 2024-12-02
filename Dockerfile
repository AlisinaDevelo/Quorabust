# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

RUN python -m pip install --upgrade "pip>=26" \
    && pip install --no-cache-dir .

USER appuser
ENTRYPOINT ["quorabust-train"]
CMD ["--help"]
