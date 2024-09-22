#!/bin/bash
set -e

export CONDA_ENV="workshop-env"
echo "Start creating Conda Environment ${CONDA_ENV}"
conda create -n CONDA_ENV python=3.9.18
conda activate workshop-env
echo "Conda Environment ${CONDA_ENV} has been created"