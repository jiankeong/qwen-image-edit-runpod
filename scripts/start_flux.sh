#!/bin/sh
set -eu
if [ "${USE_MOCK_PIPELINE:-0}" != "1" ]; then
    python /opt/flux/bootstrap_flux.py
fi
exec /start.sh
