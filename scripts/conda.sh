#!/bin/bash
set -e

CONDA="Miniconda3-latest-Linux-x86_64.sh"

# conda
mkdir -p ${HOME}/Downloads
[ -f "${HOME}/${CONDA}" ] || wget -O ${HOME}/${CONDA} "https://repo.anaconda.com/miniconda/${CONDA}"
sh ${HOME}/${CONDA} -b -u
${HOME}/miniconda3/bin/conda init
[ -f "${HOME}/${CONDA}" ] || touch ${HOME}/.zshrc
. ${HOME}/.zshrc

# related libraries
pip install httpie flake8 black isort autoflake ipython
mkdir -p ${HOME}/.config
cp flake8 ${HOME}/.config/