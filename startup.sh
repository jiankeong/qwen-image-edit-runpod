#!/bin/sh
set -eu

# Mounted volumes hide directories created during image build.
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"
exec python /workspace/handler.py
