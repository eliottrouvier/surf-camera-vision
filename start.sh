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

if [ ! -d ".venv" ]; then
    echo "📦 Création de l'environnement virtuel (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import streamlit, ultralytics, cv2, yaml" &> /dev/null; then
    echo "⚡ Installation des dépendances..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "🚀 Lancement de l'application..."
python -m streamlit run app.py
