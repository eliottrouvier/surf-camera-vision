@echo off
REM ==============================================================================
REM 🏄 Surf-Camera-Vision — Lanceur 1-Clic Automatique Windows
REM Double-cliquez sur ce fichier pour lancer l'application !
REM ==============================================================================

cd /d "%~dp0"
title Surf-Camera-Vision Studio

echo ==================================================================
echo   Surf-Camera-Vision - Initialisation du Studio PTZ
echo ==================================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    echo Telechargez Python sur https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [*] Creation de l'environnement virtuel (.venv)...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

python -c "import streamlit, ultralytics, cv2, yaml" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [*] Installation des dependances... (une seule fois)
    python -m pip install --upgrade pip
    pip install -r requirements.txt
)

echo.
echo [OK] Lancement du Studio Web Streamlit...
echo Ouverture automatique dans votre navigateur...
echo.

python -m streamlit run app.py
pause
