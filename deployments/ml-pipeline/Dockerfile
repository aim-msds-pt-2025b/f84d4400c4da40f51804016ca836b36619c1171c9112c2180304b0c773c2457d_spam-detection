FROM python:3.12-slim
# copy uv binaries from distroless uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# I'm following uv docker example here: 
# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"

RUN python3 -m nltk.downloader stopwords

COPY src/ /app/src/

CMD ["python", "-m", "src.run_pipeline"]