#!/bin/bash
set -e

# Move into the scripts directory
cd scripts || { echo "scripts directory not found! Exiting."; exit 1; }

# echo "Start setting up the environment"

# # Iterate over script files and execute them
# for cmd in base.sh py.sh fish.sh; do
#     if [ -f "$cmd" ]; then
#         echo "Running $cmd..."
#         bash "$cmd"
#     else
#         echo "Warning: $cmd not found, skipping..."
#     fi
# done

# Return to the parent directory
cd ..

source ~/.bashrc

# Activate the Conda base environment
echo "Activating Conda base environment..."
conda activate base

echo "Environment setup complete! Enjoy :-)"
