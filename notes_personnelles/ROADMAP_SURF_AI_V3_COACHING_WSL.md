# 🏄 Feuille de Route & Idées Futures — Surf-AI V3 : Coaching Multimodal & Modèle de Notation WSL

Ce document consigne les axes d'évolution stratégiques du projet Surf-Camera-Vision formulés par l'utilisateur pour les prochaines étapes de recherche et développement.

---

## 🎯 Axe 1 : Coach Virtuel IA via LLM Multimodal Vision (Gemini 2.5 Flash / Omni)

### 1. Constat & Objectif
- L'inférence biomécanique actuelle extrait des métriques chiffrées précises :
  - Angle de flexion des genoux ($\theta_{\text{knee}}$ en degrés)
  - Inclinaison du torse / prise de carre ($\theta_{\text{lean}}$ en degrés)
  - Ratio de compression athlétique
  - Vitesse réelle de la planche (km/h) et positionnement sur la vague
- **Objectif** : Transformer ces données brutes en un **débriefing de coaching personnalisé** rédigé en langage naturel par un "Coach Pro de Surf Virtuel" alimenté par un LLM multimodal avec vision native (ex. Gemini Flash Multimodal / Video Understanding).

### 2. Architecture Envisagée
```
[ Clip Vidéo du Ride (Split-Screen / 9:16) ] + [ Télémétrie Temporelle JSON ]
                              │
                              ▼
        ┌───────────────────────────────────────────────┐
        │        LLM Vision Multimodal (Gemini API)      │
        │   Système Prompt : Coach Surf Élite / CT     │
        └───────────────────────────────────────────────┘
                              │
                              ▼
    1. Analyse du Surfing : Ce que le surfeur a fait (timing, ligne, posture)
    2. Diagnostic des Défauts : Genoux trop tendus (174°), manque de relance
    3. Prescription d'Exercices : Exercices à terre (carver skateboard) et
       consignes pour la prochaine session (regard vers la section, compression 100°)
```

### 3. Exemple de Restitution UI
- Onglet dédié dans le studio Streamlit : **"🏄 Analyse du Coach IA"**
- Carte interactive avec :
  - **Points forts** : Bonne lecture de vague au take-off, glisse stable.
  - **Points bloquants** : Posture trop droite (manque d'amorti et de relance), absence de flexion avant la section creuse.
  - **Le Conseil Clé** : *"Sur cette vague de 3ft, abaisse ton centre de gravité de 25 cm dès le bas de vague pour pomper et prendre 5 km/h de vitesse."*

---

## 🏆 Axe 2 : Entraînement de Modèles de Notation sur Vidéos Officielles WSL

### 1. Constat & Objectif
- Dans les compétitions de surf officielles (WSL Championship Tour & Challenger Series), un panel de 5 juges attribue une note entre 0.0 et 10.0 en appliquant l'article 133 du règlement :
  - *Engagement et degré de difficulté*
  - *Manœuvres innovantes et progressives*
  - *Combinaison de manœuvres majeures*
  - *Vitesse, puissance et flow (Speed, Power & Flow)*
- **Objectif** : Évaluer et entraîner un modèle de prédiction de note WSL (Régression Multimodale / Video-Transformer) sur des centaines de vagues scorées lors des compétitions officielles WSL pour vérifier si l'IA converge vers les mêmes notes que les juges humains.

### 2. Méthodologie d'Implémentation
1. **Constitution du Dataset de Compétition** :
   - Vidéos HD des heats WSL (J-Bay, Teahupoo, Trestles, Pipeline, Peniche, Hossegor).
   - Métadonnées associées : note moyenne officielle, notes individuelles des 5 juges, catégorie de vague, conditions maritimes.
2. **Extraction des Signaux Prédicteurs** :
   - Features cinématiques : Vitesse d'entrée en courbe, décélération/réaccélération en sortie de manœuvre, amplitude verticale sur la vague, volume d'eau déplacé (spray segmentation).
   - Features biomécaniques : Flexion, rotation du buste, vitesse angulaire lors du snap/carve.
3. **Modélisation & Évaluation** :
   - Entraînement d'un modèle de régression (XGBoost sur features ou réseau temporel GRU/Transformer sur les embeddings vidéo).
   - Mesure de corrélation (Pearson $r$, MAE) entre la note IA et la note des juges WSL.
   - Matrice de sévérité : reproduction du barème punitif officiel (0.0-1.9 Poor, 2.0-3.9 Fair, 4.0-5.9 Good, 6.0-7.9 Very Good, 8.0-10.0 Excellent).

---

> [!NOTE]
> Ces axes de développement seront abordés et interviewés en détail (`/grill-me`) lors d'une étape dédiée après la fiabilisation complète du moteur de détection et des métriques physiques V3.
