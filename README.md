# 🏄 Surf-Camera-Vision : Autonomous Virtual PTZ Cameraman

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5c3ee8.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Transforme une caméra fixe (iPhone/4K posé sur un trépied sur la plage) en cameraman virtuel intelligent (Digital PTZ) qui traque, cadre avec les règles cinématiques et découpe automatiquement chaque vague surfée.**

![Surf-Camera-Vision Hero Banner](./assets/surf_hero_banner.png)

![Démonstration Animée du Cameraman Virtuel PTZ](./assets/virtual_cameraman_ptz_demo.gif)

---

## 📐 Modélisation & Formulation Algorithmique

Le pipeline repose sur trois composantes mathématiques simples pour le suivi et le cadrage autonome en milieu marin :

### 1. Détection du Take-Off (Classification de Posture & Cinématique)
La transition entre la phase de rame et le surf debout est caractérisée par une inversion géométrique brutale de la boîte englobante ($w/h$) corrélée à une accélération locale :

$$\text{Rame (Paddling)} : \frac{w}{h} > 1.10 \quad\longrightarrow\quad \text{Take-Off (Riding)} : \frac{w}{h} \le 0.90 \quad \text{et} \quad \|\vec{v}\| \ge v_{\text{seuil}}$$

Un tampon circulaire conserve en continu 2 secondes de pré-roll afin de garantir la capture complète de la prise de vague dès les premiers coups de rame.

---

### 2. Cadrage Virtuel & Lead-Room (Règle des Tiers)
Pour reproduire le cadrage d'un cadreur professionnel et éviter de filmer l'écume morte derrière le surfeur, le centre de la caméra virtuelle est projeté en avant du mouvement selon le vecteur vitesse $\vec{v}$ :

$$c_x^{\text{cible}} = c_x^{\text{surfeur}} + \frac{v_x}{\|\vec{v}\|} \cdot (W_{\text{crop}} \times \lambda_{\text{lead}})$$

Le surfeur est ainsi naturellement maintenu sur le tiers arrière du cadre, réservant les deux tiers de l'espace visuel au déroulement de la vague.

![Transformation Automatique : Du Plan Large aux Formats Cinéma et Mobile](./assets/comparison_wide_vs_ptz.png)

---

### 3. Amorti Cinématique (Lissage Temporel Exponentiel)
Afin d'émuler l'inertie d'une rotule fluide de trépied et de filtrer le bruit haute fréquence des détections YOLO, les coordonnées et dimensions du cadrage $(c_x, c_y, w, h)$ sont filtrées par lissage exponentiel (EMA) :

$$C_t = (1 - \alpha) \cdot C_{t-1} + \alpha \cdot C_t^{\text{cible}} \quad \text{avec } \alpha \approx 0.10$$

Ce compromis élimine tout tremblement numérique tout en conservant la réactivité nécessaire lors des virages appuyés (*bottom turns*, *cutbacks*).

---

## 🏗️ Architecture du Pipeline

```mermaid
flowchart LR
    Video["📹 Vidéo Fixe (Plage)"] --> Detector["🎯 Détecteur YOLOv8"]
    Detector --> Tracker["🧭 Tracker IoU + Vitesse"]
    Tracker --> StateMachine["⚡ Détecteur Take-Off\n(Rame -> Surf -> Sortie)"]
    StateMachine --> Cameraman["🎬 Cameraman Virtuel PTZ\n(Lead-Room + Lissage EMA)"]
    Cameraman --> Clips["✂️ Export Multi-Formats\n• 16:9 YouTube\n• 9:16 Reels / TikTok"]

    style Video fill:#1e293b,color:#fff,stroke:#3b82f6
    style Detector fill:#0284c7,color:#fff
    style Tracker fill:#0d9488,color:#fff
    style StateMachine fill:#ca8a04,color:#fff
    style Cameraman fill:#7c3aed,color:#fff
    style Clips fill:#16a34a,color:#fff
```

---

## ⚡ Démarrage Rapide

### 1. Installation
```bash
git clone https://github.com/eliottrouvier/surf-camera-vision.git
cd surf-camera-vision
pip install -r requirements.txt
```

### 2. Démo en 1 Clic
```bash
python3 demo.py
```
*Télécharge automatiquement une séquence réelle 1080p, exécute le tracking et exporte les clips découpés en 16:9 et 9:16 dans `outputs/`.*

---

## 🖥️ Utilisation

### Application Web Interactive (Streamlit)
```bash
streamlit run app.py
```
Interface moderne pour déposer vos vidéos, ajuster interactivement le zoom, la Lead-Room et télécharger les clips découpés.

### Ligne de Commande (CLI Batch)
```bash
# Export cinématique 16:9 (YouTube)
python3 track_surf.py --input ma_session.mp4 --aspect 16:9 --output-dir outputs/

# Export vertical 9:16 (Instagram Reels / TikTok) + Vidéo Double Écran (Side-by-Side 1080p)
python3 track_surf.py --input ma_session.mp4 --aspect 9:16 --lead-room 0.35 --zoom-scale 4.0
```

---

## 🧪 Tests Unitaires

```bash
pytest tests/ -v
```
Vérifie le respect des bordures (zéro bande noire), la transition de la machine à états et les équations de cadrage cinématique.

---

## 📂 Organisation du Code

```
surf-camera-vision/
├── app.py                     # Application Web Streamlit
├── demo.py                    # Démo 1-clic autonome
├── track_surf.py              # CLI batch
├── surf_camera/
│   ├── detector.py            # YOLOv8 (surfeur & planche)
│   ├── tracker.py             # Spatio-temporal tracker & vitesse
│   ├── state_machine.py       # Classification Rame -> Take-Off -> Surf
│   ├── cameraman.py           # Digital PTZ (Lead Room & lissage)
│   ├── pipeline.py            # Moteur d'assemblage & encodage H.264
│   └── utils.py               # HUD télémétrique & vidéo I/O
└── tests/                     # Suite de tests automatisés pytest
```

---

## 📄 Licence

Distribué sous licence **MIT**. Voir [`LICENSE`](LICENSE).
