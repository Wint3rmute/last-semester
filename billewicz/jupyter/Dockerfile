FROM archlinux:latest

RUN pacman -Syu --noconfirm poetry

RUN useradd -m jupyter
USER jupyter
WORKDIR /home/jupyter
RUN poetry new notebook
WORKDIR /home/jupyter/notebook
RUN echo "Only the contents of the /src directory are persisted in this docker container!" > DO_NOT_CREATE_FILES_HERE.txt
RUN poetry add jupyterlab
RUN poetry add pandas scikit-learn matplotlib numpy


