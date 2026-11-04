#!/usr/bin/env bash
# ==============================================================================
# 🏄 Surf-Camera-Vision — Lanceur Linux / Terminal
# Usage : ./start.sh
# ==============================================================================

cd "$(dirname "$0")" || exit 1

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé."
    exit 1
fi

PYTHON_BIN=""
if python3 -c "import streamlit, ultralytics, cv2, torch" &> /dev/null; then
    PYTHON_BIN="python3"
elif [ -f "$HOME/.virtualenvs/surf-camera-vision/bin/python" ]; then
    PYTHON_BIN="$HOME/.virtualenvs/surf-camera-vision/bin/python"
else
    VENV_DIR="$HOME/.virtualenvs/surf-camera-vision"
    mkdir -p "$HOME/.virtualenvs"
    python3 -m venv --system-site-packages "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR"
    PYTHON_BIN="$VENV_DIR/bin/python"
    if ! "$PYTHON_BIN" -c "import streamlit, ultralytics, cv2, torch" &> /dev/null; then
        "$PYTHON_BIN" -m pip install -r requirements.txt
    fi
fi

echo "🚀 Démarrage instantané du Studio Web Streamlit..."
(sleep 1.2 && (open "http://localhost:8501" 2>/dev/null || xdg-open "http://localhost:8501" 2>/dev/null)) &
exec "$PYTHON_BIN" -m streamlit run app.py
