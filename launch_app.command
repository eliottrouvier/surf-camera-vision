#!/usr/bin/env bash
# ==============================================================================
# 🏄 Surf-Camera-Vision — Lanceur 1-Clic Automatique macOS
# Double-cliquez sur ce fichier dans le Finder pour lancer l'application !
# ==============================================================================

# Se placer dans le dossier du projet
cd "$(dirname "$0")" || exit 1

echo "=================================================================="
echo "  🏄 Surf-Camera-Vision — Initialisation du Studio Studio PTZ     "
echo "=================================================================="
echo ""

# Vérifier la présence de Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas détecté sur votre Mac."
    echo "👉 Installez Python via https://www.python.org/downloads/ ou 'brew install python'."
    read -p "Appuyez sur Entrée pour quitter..."
    exit 1
fi

# Créer l'environnement virtuel .venv s'il n'existe pas
if [ ! -d ".venv" ]; then
    echo "📦 Création de l'environnement virtuel (.venv)..."
    python3 -m venv .venv
fi

# Activer l'environnement virtuel
source .venv/bin/activate

# Vérifier si les dépendances principales sont installées
if ! python -c "import streamlit, ultralytics, cv2, yaml" &> /dev/null; then
    echo "⚡ Installation des dépendances (ultralytics, streamlit, opencv, torch)..."
    echo "   (Cette étape ne se produit que lors du premier lancement)"
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo ""
echo "🚀 Lancement du Studio Web Streamlit..."
echo "👉 L'application va s'ouvrir automatiquement dans votre navigateur par défaut."
echo "   (Appuyez sur Ctrl+C dans cette fenêtre pour arrêter l'application)."
echo ""

# Lancement de Streamlit
python -m streamlit run app.py
