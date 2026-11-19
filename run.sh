#!/bin/sh
# Mac/Linux: open Paper Brain in your browser.   sh run.sh
# Press Ctrl+C in this window to stop it.
cd "$(dirname "$0")" || exit 1

if [ ! -x .venv/bin/python ]; then
    echo "No .venv found. Set it up first (see 'Run it' in README.md):"
    echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python train.py"
    exit 1
fi

exec .venv/bin/python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
