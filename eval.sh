#!/bin/bash
set -e

# uv sync
MODE=EVAL uv run src/main.py
