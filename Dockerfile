FROM ghcr.io/astral-sh/uv:debian

WORKDIR /app
COPY ./pyproject.toml .
COPY ./uv.lock .
COPY ./README.md .
COPY ./.python-version .
COPY ./pocket_tts_timestamped ./pocket_tts_timestamped

RUN uv run pocket-tts-timestamped --help

ENTRYPOINT ["uv", "run", "pocket-tts-timestamped"]
