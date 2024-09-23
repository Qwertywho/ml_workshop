#!/bin/bash
set -e

CONDA="Miniconda3-latest-Linux-x86_64.sh"

# conda
mkdir -p ${HOME}/Downloads
[ -f "${HOME}/${CONDA}" ] || wget -O ${HOME}/${CONDA} "https://repo.anaconda.com/miniconda/${CONDA}"
sh ${HOME}/${CONDA} -b -u
${HOME}/miniconda3/bin/conda init
[ -f "${HOME}/${CONDA}" ] || touch ${HOME}/.bashrc
. ${HOME}/.bashrc

# related libraries
${HOME}/miniconda3/bin/conda install -y -c conda-forge bat fd-find
pip install httpie flake8 black isort autoflake ipython
mkdir -p ${HOME}/.config
cp flake8 ${HOME}/.config/