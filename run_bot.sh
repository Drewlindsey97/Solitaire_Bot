#!/bin/bash
# Launch the Solitaire bot with the prepared virtualenv and local ffmpeg.
# Usage: ./run_bot.sh [bot args]   e.g. ./run_bot.sh --sim Gameplay/frame_0100.png
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
exec /home/cpt/Solitaire_Bot_venv/bin/python solitaire_auto_bot.py "$@"
