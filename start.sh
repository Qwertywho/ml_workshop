#!/bin/bash
#!/bin/bash
set -e

# Move into the scripts directory
cd scripts || { echo "scripts directory not found! Exiting."; exit 1; }

echo "Start setting up the environment"

# Iterate over script files and execute them
for cmd in base.sh py.sh fish.sh; do
    if [ -f "$cmd" ]; then
        echo "Running $cmd..."
        bash "$cmd"
    else
        echo "Warning: $cmd not found, skipping..."
    fi
done

# Return to the parent directory
cd .. || { echo "Failed to return to the parent directory! Exiting."; exit 1; }

# Activate the Conda base environment
if command -v conda >/dev/null 2>&1; then
    echo "Activating Conda base environment..."
    conda activate base
else
    echo "Conda not found! Please ensure Conda is installed and accessible."
    exit 1
fi

echo "Environment setup complete! Enjoy :-)"
