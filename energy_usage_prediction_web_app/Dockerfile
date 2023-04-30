FROM archlinux:latest

RUN pacman -Syu --noconfirm poetry

WORKDIR /app
COPY poetry.lock .
COPY pyproject.toml .
RUN poetry install --no-root

COPY . /app

