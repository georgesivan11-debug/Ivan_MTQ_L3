# 🔍 Détection de Fraude par Carte de Crédit — Projet ML

Application complète de Machine Learning pour la détection de fraudes bancaires avec interface Streamlit.

## 📋 Description

Ce projet implémente un système de détection de fraude par carte de crédit. Il identifie si une transaction est **frauduleuse (Classe 1)** ou **légitime (Classe 0)** à partir de ses caractéristiques. Le principal défi est le fort déséquilibre de classes (moins de 6% de fraudes).

## 🎯 Fonctionnalités

- ✅ **Analyse exploratoire** complète des données
- 🤖 **5 modèles de ML** testés et comparés (KNN, DT, RF, GB, LR)
- 🔧 **Optimisation des hyperparamètres** avec GridSearchCV
- ⚖️ **Gestion du déséquilibre** par oversampling
- 🎨 **Dashboard interactif** avec Streamlit
- 📊 **Visualisations** avancées (ROC, Confusion Matrix, Feature Importance)
- 📂 **Prédiction en masse** sur fichier CSV

## 🛠️ Technologies utilisées

- **Python 3.8+**
- **Scikit-learn** — Machine Learning
- **Pandas & NumPy** — Manipulation de données
- **Matplotlib & Seaborn** — Visualisation
- **Streamlit** — Interface web interactive
- **Pickle** — Sauvegarde du modèle

## 📦 Installation

1. **Cloner le repository :**
```bash
git clone https://github.com/votre-username/fraude-detection.git
cd fraude-detection
```

2. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

### 1️⃣ Entraîner le modèle

```bash
python tp_fraude_complet.py
```

Cela va :
- Charger et analyser les données
- Entraîner 5 modèles différents
- Optimiser les hyperparamètres (GridSearchCV)
- Sauvegarder le meilleur modèle (`creditcard.pkl`)
- Générer des graphiques de résultats

### 2️⃣ Lancer le Dashboard Streamlit

```bash
streamlit run streamlit_app.py
```

L'application sera accessible sur `http://localhost:8501`

## 📁 Structure du projet

```
fraude-detection/
│
├── tp_fraude_complet.py     ← Script principal d'entraînement
├── streamlit_app.py         ← Dashboard Streamlit (fichier principal)
│
├── creditcard.csv           ← Dataset
├── creditcard.pkl           ← Modèle entraîné (généré)
│
├── requirements.txt         ← Dépendances Python
├── README.md                ← Ce fichier
├── .gitignore               ← Fichiers à ignorer par Git
│
├── deploy.sh                ← Script déploiement Linux/Mac
└── deploy.bat               ← Script déploiement Windows
```

## 📊 Résultats du modèle (Random Forest Optimisé)

| Métrique   | Valeur  |
|------------|---------|
| F1-Score   | ~0.95+  |
| AUC-ROC    | ~0.99+  |
| Rappel     | ~0.95+  |
| Précision  | ~0.95+  |

## 🎓 Contexte académique

Ce projet constitue la **Partie 3** du TP2 d'Introduction à l'Intelligence Artificielle.

**LICENCE MTQ 3ème année (S6) — Année académique 2025-2026**  
**Institut Universitaire Saint Jean du Cameroun**  
Par Stéphane C. K. TÉKOUABOU (PhD & Ing.)

---

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !**
