import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report, f1_score,
    roc_auc_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Détection de Fraude 🔍",
    page_icon="🔍",
    layout="wide"
)

# Titre principal
st.title("🔍 Détection de Fraude par Carte de Crédit")
st.markdown("---")

# ============================================================
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# ============================================================
@st.cache_resource
def load_model():
    try:
        if os.path.exists('creditcard.pkl'):
            with open('creditcard.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

@st.cache_data
def load_data():
    try:
        for path in ['creditcard.csv', 'CreditCard.csv', 'CREDITCARD.csv']:
            if os.path.exists(path):
                return pd.read_csv(path)
        st.warning("Fichier creditcard.csv non trouvé.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

@st.cache_data
def preparer_donnees(df):
    X = df.drop('Class', axis=1).copy()
    y = df['Class']
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time']   = scaler.fit_transform(X[['Time']])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler

model = load_model()
df    = load_data()

if df is not None:
    X_train, X_test, y_train, y_test, scaler = preparer_donnees(df)
    feature_cols = X_test.columns.tolist()
else:
    scaler = None
    feature_cols = []

# ============================================================
# SIDEBAR — NAVIGATION
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisissez une page :",
    ["🏠 Accueil", "🔮 Prédiction", "📊 Analyse des données",
     "📈 Performances du modèle", "📂 Prédiction par fichier", "ℹ️ À propos"]
)

# ============================================================
# PAGE ACCUEIL
# ============================================================
if page == "🏠 Accueil":
    st.header("Bienvenue sur l'application de Détection de Fraude !")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📖 À propos du projet")
        st.write("""
        Cette application utilise le **Machine Learning** pour détecter les transactions
        frauduleuses par carte de crédit parmi des transactions normales.

        Le problème est une **classification binaire** :
        - **Classe 0** : Transaction légitime (normale)
        - **Classe 1** : Transaction frauduleuse

        Le principal défi est le **fort déséquilibre de classes** :
        les fraudes représentent moins de 6% des transactions.
        """)

    with col2:
        st.subheader("🎯 Fonctionnalités")
        st.write("""
        - ✅ Prédiction individuelle en temps réel
        - 📊 Exploration interactive des données
        - 📈 Métriques complètes (F1, AUC-ROC, Rappel)
        - 🔍 Matrice de confusion & courbe ROC
        - 📂 Prédiction en masse sur fichier CSV
        - 🤖 Modèle Random Forest optimisé
        """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if model is not None:
            st.success("✅ Modèle chargé avec succès !")
        else:
            st.warning("⚠️ Modèle non disponible. Exécutez `tp_fraude_complet.py` d'abord.")

    with col2:
        if df is not None:
            st.success(f"✅ Dataset chargé : {len(df):,} transactions")
        else:
            st.error("❌ Dataset non disponible")

    with col3:
        if df is not None:
            n_fraud = df['Class'].sum()
            st.info(f"⚠️ Fraudes détectées : {n_fraud} ({n_fraud/len(df)*100:.2f}%)")

    st.markdown("---")
    st.info("👈 Utilisez le menu à gauche pour naviguer entre les différentes pages")

# ============================================================
# PAGE PRÉDICTION
# ============================================================
elif page == "🔮 Prédiction":
    st.header("Prédiction sur une Transaction")

    if model is None or scaler is None:
        st.error("❌ Modèle non disponible. Veuillez d'abord entraîner le modèle.")
        st.info("Exécutez le fichier `tp_fraude_complet.py` pour créer les fichiers nécessaires.")
        st.code("python tp_fraude_complet.py", language="bash")
        st.stop()

    st.write("Entrez les caractéristiques d'une transaction pour obtenir une prédiction :")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💳 Paramètres de la transaction")
        amount = st.number_input("💰 Montant (€)", min_value=0.0, max_value=25000.0, value=150.0, step=10.0)
        time   = st.number_input("⏱️ Temps (secondes depuis 1ère transaction)", min_value=0.0, value=50000.0, step=1000.0)

        st.markdown("**Variables PCA (V1 à V10) :**")
        cols_input = st.columns(2)
        v_vals = {}
        for i in range(1, 11):
            with cols_input[(i-1) % 2]:
                v_vals[f'V{i}'] = st.number_input(f"V{i}", value=0.0, step=0.1,
                                                    key=f"v{i}", format="%.3f")

        st.markdown("---")
        use_example = st.toggle("🎲 Charger un exemple aléatoire du dataset")
        if use_example and df is not None:
            ex_type = st.radio("Type d'exemple :", ["Normale", "Fraude"], horizontal=True)
            val = 0 if ex_type == "Normale" else 1
            row = df[df['Class'] == val].sample(1, random_state=np.random.randint(0, 200))
            amount = float(row['Amount'].values[0])
            time   = float(row['Time'].values[0])
            for i in range(1, 11):
                v_vals[f'V{i}'] = float(row[f'V{i}'].values[0])
            st.success(f"Exemple « {ex_type} » chargé !")

    with col2:
        st.subheader("🎯 Résultat de la prédiction")

        if st.button("🔮 Analyser cette transaction", type="primary", use_container_width=True):
            input_data = {'Time': time, 'Amount': amount}
            for i in range(1, 11):
                input_data[f'V{i}'] = v_vals[f'V{i}']

            input_df = pd.DataFrame([input_data])[feature_cols]
            input_df['Amount'] = scaler.transform(input_df[['Amount']])
            input_df['Time']   = scaler.transform(input_df[['Time']])

            prediction = model.predict(input_df)[0]
            proba      = model.predict_proba(input_df)[0]

            if prediction == 1:
                st.error("🚨 TRANSACTION FRAUDULEUSE DÉTECTÉE !")
            else:
                st.success("✅ TRANSACTION LÉGITIME")

            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.metric("Probabilité Normale",  f"{proba[0]*100:.1f}%")
            c2.metric("Probabilité Fraude",   f"{proba[1]*100:.1f}%")

            # Graphique probabilités
            fig, ax = plt.subplots(figsize=(8, 3))
            colors = ['#2ecc71', '#e74c3c']
            bars = ax.barh(['Normale', 'Fraude'], proba, color=colors)
            ax.set_xlim([0, 1])
            ax.set_xlabel('Probabilité')
            ax.set_title('Probabilités par classe', fontweight='bold')
            for i, (bar, v) in enumerate(zip(bars, proba)):
                ax.text(v + 0.02, i, f'{v:.2%}', va='center', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("---")
            st.markdown(f"**Montant :** {amount:.2f} €")
            st.markdown(f"**Décision :** {'🔴 FRAUDE' if prediction == 1 else '🟢 NORMALE'}")
            st.markdown(f"**Confiance :** {max(proba)*100:.1f}%")

# ============================================================
# PAGE ANALYSE DES DONNÉES
# ============================================================
elif page == "📊 Analyse des données":
    st.header("Exploration et Visualisation des Données")

    if df is None:
        st.error("❌ Dataset non disponible.")
        st.stop()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total transactions", f"{len(df):,}")
    col2.metric("Fraudes", f"{df['Class'].sum()}")
    col3.metric("Taux de fraude", f"{df['Class'].mean()*100:.2f}%")
    col4.metric("Variables", f"{df.shape[1]-1}")

    st.markdown("---")

    # Distribution des classes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Répartition des classes")
        counts = df['Class'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['Normale (0)', 'Fraude (1)'], counts.values,
                      color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f'{val:,}\n({val/len(df)*100:.2f}%)', ha='center', fontweight='bold')
        ax.set_ylabel("Nombre de transactions")
        ax.set_title("Distribution des classes", fontweight='bold')
        ax.set_ylim(0, counts.max() * 1.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Distribution des montants")
        fig, ax = plt.subplots(figsize=(6, 4))
        df[df['Class']==0]['Amount'].clip(upper=500).hist(
            bins=50, ax=ax, color='#2ecc71', alpha=0.7, label='Normale', density=True)
        df[df['Class']==1]['Amount'].clip(upper=500).hist(
            bins=30, ax=ax, color='#e74c3c', alpha=0.7, label='Fraude', density=True)
        ax.set_xlabel("Montant (€)")
        ax.set_ylabel("Densité")
        ax.set_title("Distribution des montants par classe", fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Boxplots features
    st.subheader("Distribution des features V1–V10 par classe")
    features_v = [f'V{i}' for i in range(1, 11)]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()
    for i, feat in enumerate(features_v):
        data = [df[df['Class']==0][feat].values, df[df['Class']==1][feat].values]
        bp = axes[i].boxplot(data, patch_artist=True,
                              medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        axes[i].set_title(feat, fontweight='bold')
        axes[i].set_xticklabels(['Normal', 'Fraude'], fontsize=8)
    plt.suptitle("Distribution des features V1–V10 par classe", fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Variable à choisir
    st.subheader("📌 Analyse d'une variable au choix")
    selected_var = st.selectbox("Choisissez une variable :", features_v + ['Amount', 'Time'])

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        df[df['Class']==0][selected_var].hist(bins=40, ax=ax, color='#2ecc71', alpha=0.7, label='Normale', density=True)
        df[df['Class']==1][selected_var].hist(bins=20, ax=ax, color='#e74c3c', alpha=0.7, label='Fraude', density=True)
        ax.set_title(f"Distribution de {selected_var}", fontweight='bold')
        ax.set_xlabel(selected_var)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='Class', y=selected_var, ax=ax,
                    palette={0: '#2ecc71', 1: '#e74c3c'})
        ax.set_xticklabels(['Normale', 'Fraude'])
        ax.set_title(f"{selected_var} par classe", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Matrice de corrélation
    st.subheader("🔗 Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, ax=ax,
                linewidths=0.3, annot=False, cbar_kws={'label': 'Corrélation'})
    ax.set_title("Matrice de corrélation", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 Voir les données brutes"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Affichage des 100 premières lignes sur {len(df):,}")

# ============================================================
# PAGE PERFORMANCES
# ============================================================
elif page == "📈 Performances du modèle":
    st.header("Évaluation des Performances du Modèle")

    if model is None or df is None:
        st.error("❌ Modèle ou données non disponibles.")
        st.stop()

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1    = f1_score(y_test, y_pred)
    auc_s = roc_auc_score(y_test, y_proba)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred)
    acc   = model.score(X_test, y_test)

    # Métriques
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy",   f"{acc:.4f}")
    col2.metric("F1-Score",   f"{f1:.4f}")
    col3.metric("AUC-ROC",    f"{auc_s:.4f}")
    col4.metric("Précision",  f"{prec:.4f}")
    col5.metric("Rappel",     f"{rec:.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matrice de confusion")
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normale', 'Fraude'],
                    yticklabels=['Normale', 'Fraude'],
                    linewidths=1, annot_kws={'size': 16, 'weight': 'bold'})
        ax.set_xlabel("Prédit", fontweight='bold')
        ax.set_ylabel("Réel", fontweight='bold')
        ax.set_title("Matrice de Confusion", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        vp, fn, fp = cm[1,1], cm[1,0], cm[0,1]
        st.markdown(f"- ✅ **Fraudes correctement détectées** : {vp}")
        st.markdown(f"- ❌ **Fraudes manquées (Faux Négatifs)** : {fn} ⚠️")
        st.markdown(f"- ⚠️ **Fausses alarmes (Faux Positifs)** : {fp}")

    with col2:
        st.subheader("Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'Random Forest (AUC = {auc_val:.4f})')
        ax.plot([0,1],[0,1], 'k--', lw=1.5, label='Aléatoire (AUC = 0.5000)')
        ax.fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
        ax.set_xlabel("Taux de Faux Positifs (FPR)", fontweight='bold')
        ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontweight='bold')
        ax.set_title("Courbe ROC", fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Importance des variables
    st.subheader("🔑 Importance des variables")
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:12]
    feat_names = feature_cols
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_imp = ['#e74c3c' if i < 3 else '#3498db' for i in range(len(idx))]
    ax.bar([feat_names[i] for i in idx], importances[idx], color=colors_imp, edgecolor='black')
    ax.set_ylabel("Importance (Gini)")
    ax.set_title("Top 12 variables les plus importantes", fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 Rapport de classification complet"):
        report = classification_report(y_test, y_pred, target_names=['Normale', 'Fraude'])
        st.code(report, language='text')

# ============================================================
# PAGE PRÉDICTION PAR FICHIER
# ============================================================
elif page == "📂 Prédiction par fichier":
    st.header("Prédiction en masse sur un fichier CSV")

    if model is None or scaler is None:
        st.error("❌ Modèle non disponible.")
        st.stop()

    st.info("📁 Importez un fichier CSV contenant des transactions pour obtenir des prédictions en masse.")
    uploaded = st.file_uploader("Choisissez un fichier CSV", type=['csv'])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"✅ Fichier chargé : {len(df_up):,} lignes, {df_up.shape[1]} colonnes")
            st.dataframe(df_up.head(), use_container_width=True)

            missing_cols = [c for c in feature_cols if c not in df_up.columns]
            if missing_cols:
                st.error(f"❌ Colonnes manquantes : {missing_cols}")
            else:
                X_up = df_up[feature_cols].copy()
                X_up['Amount'] = scaler.transform(X_up[['Amount']])
                X_up['Time']   = scaler.transform(X_up[['Time']])

                preds  = model.predict(X_up)
                probas = model.predict_proba(X_up)[:, 1]

                df_up['Prédiction'] = preds
                df_up['Probabilité_Fraude'] = probas.round(4)
                df_up['Statut'] = df_up['Prédiction'].map({0: '✅ Normale', 1: '🔴 Fraude'})

                n_fraud = preds.sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total analysé", f"{len(preds):,}")
                c2.metric("Fraudes détectées", f"{n_fraud}", delta=f"{n_fraud/len(preds)*100:.2f}%")
                c3.metric("Transactions normales", f"{len(preds)-n_fraud}")

                st.dataframe(
                    df_up[['Statut', 'Probabilité_Fraude'] + feature_cols[:5]],
                    use_container_width=True
                )
                csv_out = df_up.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Télécharger les résultats", csv_out,
                                   file_name="resultats_predictions.csv",
                                   mime="text/csv", type="primary")
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.markdown("""
        #### Format attendu du fichier CSV :
        Le fichier doit contenir les colonnes : `Time`, `V1`, `V2`, ..., `V10`, `Amount`
        > 💡 Vous pouvez utiliser le fichier `creditcard.csv` fourni comme exemple.
        """)

# ============================================================
# PAGE À PROPOS
# ============================================================
elif page == "ℹ️ À propos":
    st.header("À propos de ce projet")

    st.markdown("""
    ### 🎓 Partie 3 — TP2 IIA | LICENCE MTQ S6 | IUSJ Cameroun 2025-2026

    Ce projet a été développé dans le cadre du TP2 d'Introduction à l'Intelligence Artificielle.

    #### 🛠️ Technologies utilisées :
    - **Python** : Langage de programmation
    - **Scikit-learn** : Bibliothèque de Machine Learning
    - **Pandas & NumPy** : Manipulation de données
    - **Matplotlib & Seaborn** : Visualisation
    - **Streamlit** : Interface web interactive
    - **Pickle** : Sauvegarde du modèle

    #### 📚 Dataset :
    Le dataset utilisé est inspiré du **Credit Card Fraud Detection** (Kaggle / ULB).
    - **~10 000 transactions** simulées
    - **Variables V1–V10** : composantes PCA (confidentialité)
    - **Amount** : montant de la transaction
    - **Class** : 0 = normale, 1 = fraude

    #### 🤖 Modèle utilisé :
    - **Random Forest Classifier** (méthode d'ensemble — Bagging)
    - Optimisé par **GridSearchCV** avec validation croisée stratifiée (5 folds)
    - Gestion du **déséquilibre** : Oversampling de la classe minoritaire

    #### 🎯 Résultats typiques :
    | Métrique | Valeur |
    |----------|--------|
    | F1-Score | ~0.95+ |
    | AUC-ROC  | ~0.99+ |
    | Rappel   | ~0.95+ |
    """)

    if df is not None:
        st.markdown("---")
        st.subheader("📊 Informations sur le dataset actuel")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total transactions", f"{len(df):,}")
        c2.metric("Fraudes", f"{df['Class'].sum()}")
        c3.metric("Variables explicatives", f"{df.shape[1]-1}")

    st.success("✅ Application développée avec ❤️ pour l'apprentissage du ML")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>"
    "🔍 Fraud Detector — TP2 IIA ML 2025-2026 | "
    "Développé avec Streamlit & Scikit-learn | IUSJ Cameroun"
    "</div>",
    unsafe_allow_html=True
)
