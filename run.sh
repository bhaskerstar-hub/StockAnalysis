#!/bin/bash
# Kill any running instance on port 5001 then start fresh
lsof -ti:5001 | xargs kill -9 2>/dev/null
sleep 0.5
cd "$(dirname "$0")"
python3 app.py
