#!/bin/bash
set -e

pip install httpie flake8 black isort autoflake ipython
mkdir -p ${HOME}/.config
cp flake8 ${HOME}/.config/

export CONDA_ENV="workshop-env"
echo "Start creating Conda Environment ${CONDA_ENV}"
conda create -n ${CONDA_ENV} python=3.9.18 -y
conda activate ${CONDA_ENV}
echo "Conda Environment ${CONDA_ENV} has been created"
