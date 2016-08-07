#!/usr/bin/env bash
# Create a separate Python environment for Orange and its dependencies,
# and make it the active one
virtualenv --python=python3 --system-site-packages orange3venv-recsys
source orange3venv-recsys/bin/activate

# Install the minimum required dependencies first
pip install -r requirements.txt

# Finally install Orange in editable/development mode.
pip install -e .