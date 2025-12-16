#!/bin/bash
set -e

# uv sync
MODE=TRAIN uv run src/main.py
