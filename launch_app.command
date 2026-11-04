#!/usr/bin/env bash
# ==============================================================================
# 🏄 Surf-Camera-Vision — Lanceur 1-Clic Automatique macOS
# Double-cliquez sur ce fichier dans le Finder pour lancer l'application !
# ==============================================================================

# Se placer dans le dossier du projet
cd "$(dirname "$0")" || exit 1

echo "=================================================================="
echo "  🏄 Surf-Camera-Vision — Studio d'Inférence Dédié               "
echo "=================================================================="
echo ""

# Vérifier la présence de Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas détecté sur votre Mac."
    echo "👉 Installez Python via https://www.python.org/downloads/ ou 'brew install python'."
    read -p "Appuyez sur Entrée pour quitter..."
    exit 1
fi

# Détecter le runtime Python le plus rapide déjà prêt
PYTHON_BIN=""

if python3 -c "import streamlit, ultralytics, cv2, torch" &> /dev/null; then
    PYTHON_BIN="python3"
elif [ -f "$HOME/.virtualenvs/surf-camera-vision/bin/python" ]; then
    PYTHON_BIN="$HOME/.virtualenvs/surf-camera-vision/bin/python"
else
    VENV_DIR="$HOME/.virtualenvs/surf-camera-vision"
    echo "📦 Initialisation de l'environnement dédié ($VENV_DIR)..."
    mkdir -p "$HOME/.virtualenvs"
    python3 -m venv --system-site-packages "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR"
    PYTHON_BIN="$VENV_DIR/bin/python"

    if ! "$PYTHON_BIN" -c "import streamlit, ultralytics, cv2, torch" &> /dev/null; then
        echo "⚡ Installation des dépendances légères d'inférence..."
        "$PYTHON_BIN" -m pip install -r requirements.txt
    fi
fi

echo "🚀 Démarrage instantané du Studio Web Streamlit..."
echo "👉 Ouverture automatique de votre navigateur sur http://localhost:8501..."
echo "   (Appuyez sur Ctrl+C dans cette fenêtre pour arrêter l'application)."
echo ""

# Ouverture automatique garantie du navigateur Safari / Chrome par défaut sur macOS
(sleep 1.2 && open "http://localhost:8501") &

exec "$PYTHON_BIN" -m streamlit run app.py
