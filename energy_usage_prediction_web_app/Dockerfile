FROM archlinux:latest

RUN pacman -Syu --noconfirm poetry

WORKDIR /app
COPY poetry.lock .
COPY pyproject.toml .
RUN poetry install --no-root

COPY . /app

# Run the notebook once to train & save the model
RUN poetry run ipython source_notebook.ipynb 
RUN mv /tmp/model .
