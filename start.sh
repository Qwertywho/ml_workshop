#!/bin/bash
set -e

cd scripts
echo "Start setting up the environment"

for cmd in base.sh conda.sh py.sh fish.sh
do
    bash $cmd
done

cd

echo "Enjoy :-)"