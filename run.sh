#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import nltk
nltk.download('punkt')
try:
    nltk.download('punkt_tab')
except Exception:
    pass
PY
python main.py --all
