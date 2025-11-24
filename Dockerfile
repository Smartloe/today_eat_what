FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install uv for locked installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.cargo/bin/uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY today_eat_what ./today_eat_what
COPY main.py ./main.py

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "-m", "today_eat_what"]
