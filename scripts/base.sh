#!/bin/bash
set -e

# Debian
sudo apt-get update
sudo apt-get install -y curl wget git vim build-essential git-lfs
sudo apt-get autoremove