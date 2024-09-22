#!/bin/bash
set -e

mkdir -p ${HOME}/.config
${HOME}/miniconda3/bin/conda install -c conda-forge -y fish

mkdir -p ${HOME}/.config/fish
cp config.fish ${HOME}/.config/fish/