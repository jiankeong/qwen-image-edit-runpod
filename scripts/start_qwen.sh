#!/bin/sh
set -eu

if [ "${USE_MOCK_PIPELINE:-0}" != "1" ]; then
    python /opt/qwen/bootstrap_models.py
fi

exec /start.sh
