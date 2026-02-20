# 🚀 Guide de Déploiement sur GitHub et Streamlit Cloud

## 📋 Étape 1 : Préparation des fichiers

Avant de déployer, assurez-vous d'avoir ces fichiers dans votre dossier :

```
fraude-detection/
├── tp_fraude_complet.py
├── streamlit_app.py
├── creditcard.csv
├── creditcard.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── deploy.sh
└── deploy.bat
```

## 🐙 Étape 2 : Déploiement sur GitHub

### 2.1 Créer un compte GitHub (si vous n'en avez pas)
1. Allez sur https://github.com
2. Cliquez sur "Sign up"
3. Suivez les instructions

### 2.2 Créer un nouveau repository

1. Sur GitHub, cliquez sur le bouton vert **"New"** ou **"+"** → **"New repository"**
2. Remplissez les informations :
   - **Repository name** : `fraude-detection` (ou autre nom)
   - **Description** : "Détection de fraude par carte de crédit avec ML"
   - **Public** ou **Private** : à votre choix
   - **Ne cochez PAS** "Add a README" (on a déjà le nôtre)
3. Cliquez sur **"Create repository"**

### 2.3 Initialiser Git localement

Ouvrez un terminal dans votre dossier projet et exécutez :

```bash
# Initialiser Git
git init

# Ajouter tous les fichiers
git add .

# Créer le premier commit
git commit -m "Premier commit - Projet détection de fraude ML"

# Renommer la branche en 'main'
git branch -M main

# Lier au repository GitHub (remplacez YOUR-USERNAME et YOUR-REPO)
git remote add origin https://github.com/YOUR-USERNAME/fraude-detection.git

# Pousser le code vers GitHub
git push -u origin main
```

### 2.4 Vérifier sur GitHub

Retournez sur votre page GitHub et rafraîchissez. Vous devriez voir tous vos fichiers !

## ☁️ Étape 3 : Déploiement sur Streamlit Cloud

### 3.1 Créer un compte Streamlit Cloud

1. Allez sur https://streamlit.io/cloud
2. Cliquez sur **"Sign up"**
3. Connectez-vous avec votre compte **GitHub**

### 3.2 Déployer l'application

1. Une fois connecté, cliquez sur **"New app"**
2. Remplissez les informations :
   - **Repository** : Sélectionnez `YOUR-USERNAME/fraude-detection`
   - **Branch** : `main`
   - **Main file path** : `streamlit_app.py`
   - **App URL** : Choisissez un nom (ex: `fraude-detector-iusj`)
3. Cliquez sur **"Deploy!"**

### 3.3 Important — Fichiers nécessaires dans le repo

Pour que Streamlit Cloud fonctionne, vous devez avoir dans votre repo GitHub :
- ✅ `creditcard.csv` — le dataset
- ✅ `creditcard.pkl` — le modèle entraîné
- ✅ `requirements.txt` — les dépendances

**Si `creditcard.pkl` n'est pas dans le repo**, ajoutez ce code au début de `streamlit_app.py` :

```python
import os
if not os.path.exists('creditcard.pkl'):
    os.system('python tp_fraude_complet.py')
```

### 3.4 Attendre le déploiement

- Des logs vont défiler pendant 2-3 minutes
- Si erreur, vérifiez que `creditcard.csv` est bien dans le repo
- En cas de "ModuleNotFoundError", vérifiez `requirements.txt`

## 🔄 Étape 4 : Mises à jour futures

Pour mettre à jour votre code :

```bash
git add .
git commit -m "Description des changements"
git push
```

Streamlit Cloud redéploiera automatiquement votre app !

## 🐛 Dépannage

### Problème : "ModuleNotFoundError"
**Solution** : Vérifiez que toutes les dépendances sont dans `requirements.txt`

### Problème : "FileNotFoundError: creditcard.csv"
**Solution** : Assurez-vous que `creditcard.csv` est bien dans le repo GitHub

### Problème : "FileNotFoundError: creditcard.pkl"
**Solution** : Exécutez d'abord `python tp_fraude_complet.py` et committez le `.pkl`

### Problème : Git demande un mot de passe
**Solution** : Utilisez un Personal Access Token :
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token
3. Utilisez le token comme mot de passe

## 📱 Étape 5 : Partager votre application

Une fois déployée, vous obtiendrez une URL comme :
```
https://fraude-detector-iusj.streamlit.app
```

Partagez cette URL avec qui vous voulez ! 🎉

## 🎯 Checklist finale

- [ ] `tp_fraude_complet.py` exécuté (creditcard.pkl créé)
- [ ] App testée localement (`streamlit run streamlit_app.py`)
- [ ] Code poussé sur GitHub
- [ ] `creditcard.csv` présent dans le repo
- [ ] App déployée sur Streamlit Cloud
- [ ] App testée et fonctionnelle en ligne
- [ ] URL partageable obtenue

Bon déploiement ! 🚀
