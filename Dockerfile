FROM nvcr.io/nvidia/pytorch:25.08-py3 AS base

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG DEV_computervision

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_SRC=/src \
    NO_COLOR=true \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=true \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr

# Ports for jupyter and tensorboard
EXPOSE 8888
EXPOSE 6006

RUN mkdir -p /app
WORKDIR /app

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
     uv sync --frozen --inexact