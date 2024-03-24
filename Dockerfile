FROM python:3.10-slim

ENV POETRY_VERSION=1.8.1

# Install system dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    musl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app
COPY poetry.lock pyproject.toml /app/

RUN poetry install --no-interaction

COPY models/ /app/models/
COPY src/ /app/src/

EXPOSE 3000

CMD ["poetry", "run", "serve"]