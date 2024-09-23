#!/bin/bash
set -e

CONDA="Miniconda3-latest-Linux-x86_64.sh"
CONDA_DIR="${HOME}/miniconda3"
CONDA_ENV="workshop-env"

# Ensure Downloads directory exists
mkdir -p ${HOME}/Downloads

# Download Miniconda installer if it doesn't already exist
if [ ! -f "${HOME}/Downloads/${CONDA}" ]; then
    wget -O ${HOME}/Downloads/${CONDA} "https://repo.anaconda.com/miniconda/${CONDA}"
fi

# Install Miniconda silently
sh ${HOME}/Downloads/${CONDA} -b -u -p ${CONDA_DIR}

# Initialize conda
${CONDA_DIR}/bin/conda init

# Ensure .bashrc exists
if [ ! -f "${HOME}/.bashrc" ]; then
    touch ${HOME}/.bashrc
fi

# Source .bashrc to apply conda init changes
. ${HOME}/.bashrc

# Install related libraries via conda
${CONDA_DIR}/bin/conda install -y -c conda-forge bat fd-find

# Install Python packages via pip
${CONDA_DIR}/bin/pip install httpie flake8 black isort autoflake ipython

# Create config directory if it doesn't exist and copy flake8 config
mkdir -p ${HOME}/.config
if [ -f flake8 ]; then
    cp flake8 ${HOME}/.config/
else
    echo "flake8 configuration file not found, skipping copy."
fi

# Export Conda environment variable
export CONDA_ENV="workshop-env"
echo "Start creating Conda Environment ${CONDA_ENV}"

# Create the Conda environment
${CONDA_DIR}/bin/conda create -n ${CONDA_ENV} python=3.9.18 -y

# Activate the Conda environment
echo "Conda Environment ${CONDA_ENV} has been created. Activating now..."
. ${CONDA_DIR}/bin/activate ${CONDA_ENV}

# Confirmation of activation
echo "Conda Environment ${CONDA_ENV} is now active."
