#!/bin/sh
echo "Starting on port: $PORT"
exec .venv/bin/uvicorn main:app --host 0.0.0.0 --port "$PORT"
