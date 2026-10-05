#!/bin/bash
# FringeLab launcher for macOS / Linux: double-click this file (macOS) or run
#   bash run_app.command
# The first run creates a private Python environment in .venv and installs the requirements.
cd "$(dirname "$0")" || exit 1
if [ ! -f .venv/installed.txt ]; then
    echo "First start: installing FringeLab. This takes a few minutes and happens only once."
    if ! python3 -m venv .venv \
        || ! .venv/bin/python -m pip install --upgrade pip \
        || ! .venv/bin/python -m pip install -r requirements.txt; then
        echo
        echo "Setup failed. Check that Python 3.10 or newer is installed (python.org), then start this file again."
        read -r -p "Press Enter to close."
        exit 1
    fi
    echo ok > .venv/installed.txt
fi
exec .venv/bin/python launch.py
