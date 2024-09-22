#!/bin/bash
set -e

cd scripts
echo "Start setting up the environment"

for cmd in base.sh conda.sh fish.sh
do
    bash $cmd
done

echo "Enjoy :-)"