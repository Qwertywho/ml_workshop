#!/bin/bash
set -e

# Create .config directory if it doesn't exist
mkdir -p "${HOME}/.config"

# Check if Conda is installed and install Fish shell using Conda
if [ -f "${HOME}/miniconda3/bin/conda" ]; then
    echo "Installing Fish shell via Conda..."
    "${HOME}/miniconda3/bin/conda" install -c conda-forge -y fish
else
    echo "Conda not found! Please install Conda first."
    exit 1
fi

# Ensure the Fish configuration directory exists
mkdir -p "${HOME}/.config/fish"

# Copy the Fish shell configuration file
if [ -f "config.fish" ]; then
    echo "Copying config.fish to ${HOME}/.config/fish/..."
    cp config.fish "${HOME}/.config/fish/"
else
    echo "config.fish file not found! Please ensure it exists in the current directory."
    exit 1
fi

echo "Fish shell setup complete!"
