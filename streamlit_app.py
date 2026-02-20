import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report, f1_score,
    roc_auc_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Détection de Fraude 🔍",
    page_icon="🔍",
    layout="wide"
)
st.title("🔍 Détection de Fraude par Carte de Crédit")
st.markdown("---")

# ============================================================
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# ============================================================
@st.cache_resource
def load_model():
    try:
        if not os.path.exists('creditcard.pkl'):
            return None, None, None, []
        with open('creditcard.pkl', 'rb') as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict):
            return (bundle['model'], bundle['scaler_amount'],
                    bundle['scaler_time'], bundle['feature_cols'])
        else:
            return bundle, None, None, []
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None, None, None, []

@st.cache_data
def load_data():
    for path in ['creditcard.csv', 'CreditCard.csv', 'CREDITCARD.csv']:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return None

model, scaler_amount, scaler_time, feature_cols = load_model()
df = load_data()

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
# PAGE — ACCUEIL
# ============================================================
if page == "🏠 Accueil":
    st.header("Bienvenue sur l'application de Détection de Fraude !")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📖 À propos du projet")
        st.write("""
        Cette application utilise le **Machine Learning** pour détecter
        les transactions frauduleuses par carte de crédit.

        Le problème est une **classification binaire** :
        - **Classe 0** : Transaction légitime
        - **Classe 1** : Transaction frauduleuse

        Le principal défi : **fort déséquilibre de classes**
        (moins de 6% de fraudes dans les données réelles).
        """)
    with col2:
        st.subheader("🎯 Fonctionnalités")
        st.write("""
        - ✅ Prédiction individuelle en temps réel
        - 📊 Exploration interactive des données
        - 📈 Métriques complètes (F1, AUC-ROC, Rappel)
        - 📂 Prédiction sur **n'importe quel fichier CSV**
        - 🤖 Modèle Random Forest optimisé par GridSearchCV
        """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if model is not None:
            st.success("✅ Modèle chargé avec succès !")
        else:
            st.warning("⚠️ Modèle non disponible.")
    with col2:
        if df is not None:
            st.success(f"✅ Dataset : {len(df):,} transactions")
        else:
            st.error("❌ Dataset non disponible")
    with col3:
        if df is not None and 'Class' in df.columns:
            n = df['Class'].sum()
            st.info(f"⚠️ Fraudes : {n} ({n/len(df)*100:.2f}%)")

    st.info("👈 Utilisez le menu à gauche pour naviguer entre les pages")

# ============================================================
# PAGE — PRÉDICTION INDIVIDUELLE
# ============================================================
elif page == "🔮 Prédiction":
    st.header("Prédiction sur une Transaction")

    if model is None:
        st.error("❌ Modèle non disponible. Vérifiez que creditcard.pkl est bien présent.")
        st.stop()

    if not feature_cols:
        st.error("❌ Impossible de lire les colonnes depuis creditcard.pkl.")
        st.stop()

    st.write("Entrez les caractéristiques d'une transaction pour obtenir une prédiction :")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("💳 Paramètres de la transaction")

        amount  = st.number_input("💰 Montant (€)", min_value=0.0,
                                   max_value=25000.0, value=150.0, step=10.0)
        time_v  = st.number_input("⏱️ Temps (secondes)", min_value=0.0,
                                   value=50000.0, step=1000.0)

        v_features = [c for c in feature_cols if c.startswith('V')]
        st.markdown("**Variables PCA :**")
        grid = st.columns(2)
        v_vals = {}
        for i, feat in enumerate(v_features):
            with grid[i % 2]:
                v_vals[feat] = st.number_input(
                    feat, value=0.0, step=0.1,
                    key=f"pred_{feat}", format="%.3f"
                )

        st.markdown("---")
        if df is not None and 'Class' in df.columns:
            use_ex = st.toggle("🎲 Charger un exemple aléatoire du dataset")
            if use_ex:
                ex_type = st.radio("Type :", ["Normale", "Fraude"], horizontal=True)
                classe  = 0 if ex_type == "Normale" else 1
                row = df[df['Class'] == classe].sample(
                    1, random_state=np.random.randint(0, 999)
                )
                if 'Amount' in row.columns:
                    amount = float(row['Amount'].values[0])
                if 'Time' in row.columns:
                    time_v = float(row['Time'].values[0])
                for feat in v_features:
                    if feat in row.columns:
                        v_vals[feat] = float(row[feat].values[0])
                st.success(f"Exemple « {ex_type} » chargé !")

    with col_right:
        st.subheader("🎯 Résultat de la prédiction")

        if st.button("🔮 Analyser cette transaction",
                     type="primary", use_container_width=True):
            try:
                # Construction du vecteur dans le bon ordre
                input_row = {}
                for col in feature_cols:
                    if col == 'Amount':
                        input_row[col] = amount
                    elif col == 'Time':
                        input_row[col] = time_v
                    else:
                        input_row[col] = v_vals.get(col, 0.0)

                X_in = pd.DataFrame([input_row])[feature_cols]

                # Normalisation avec les scalers sauvegardés
                if scaler_amount is not None and 'Amount' in X_in.columns:
                    X_in['Amount'] = scaler_amount.transform(X_in[['Amount']])
                if scaler_time is not None and 'Time' in X_in.columns:
                    X_in['Time'] = scaler_time.transform(X_in[['Time']])

                prediction = model.predict(X_in)[0]
                proba      = model.predict_proba(X_in)[0]

                # Résultat
                if prediction == 1:
                    st.error("🚨 TRANSACTION FRAUDULEUSE DÉTECTÉE !")
                else:
                    st.success("✅ TRANSACTION LÉGITIME")

                st.markdown("---")
                c1, c2 = st.columns(2)
                c1.metric("Probabilité Normale", f"{proba[0]*100:.1f}%")
                c2.metric("Probabilité Fraude",  f"{proba[1]*100:.1f}%")

                # Graphique probabilités
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.barh(['Normale', 'Fraude'], proba,
                        color=['#2ecc71', '#e74c3c'])
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probabilité')
                ax.set_title('Probabilités par classe', fontweight='bold')
                for i, v in enumerate(proba):
                    ax.text(v + 0.01, i, f'{v:.2%}',
                            va='center', fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown("---")
                st.markdown(f"**Montant :** {amount:.2f} €")
                st.markdown(f"**Décision :** {'🔴 FRAUDE' if prediction==1 else '🟢 NORMALE'}")
                st.markdown(f"**Confiance :** {max(proba)*100:.1f}%")

            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction : {e}")

# ============================================================
# PAGE — ANALYSE DES DONNÉES
# ============================================================
elif page == "📊 Analyse des données":
    st.header("Exploration et Visualisation des Données")

    if df is None:
        st.error("❌ Dataset non disponible.")
        st.stop()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total transactions", f"{len(df):,}")
    if 'Class' in df.columns:
        col2.metric("Fraudes",      f"{df['Class'].sum()}")
        col3.metric("Taux fraude",  f"{df['Class'].mean()*100:.2f}%")
    col4.metric("Variables",        f"{df.shape[1]}")

    st.markdown("---")

    if 'Class' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Répartition des classes")
            counts = df['Class'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(['Normale (0)', 'Fraude (1)'], counts.values,
                          color=['#2ecc71', '#e74c3c'], edgecolor='black')
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 10,
                        f'{val:,}\n({val/len(df)*100:.2f}%)',
                        ha='center', fontweight='bold')
            ax.set_ylabel("Nombre")
            ax.set_ylim(0, counts.max() * 1.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            if 'Amount' in df.columns:
                st.subheader("Distribution des montants")
                fig, ax = plt.subplots(figsize=(6, 4))
                df[df['Class']==0]['Amount'].clip(upper=500).hist(
                    bins=50, ax=ax, color='#2ecc71',
                    alpha=0.7, label='Normale', density=True)
                df[df['Class']==1]['Amount'].clip(upper=500).hist(
                    bins=30, ax=ax, color='#e74c3c',
                    alpha=0.7, label='Fraude', density=True)
                ax.set_xlabel("Montant (€)")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # Matrice de corrélation
    st.subheader("🔗 Matrice de corrélation")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=bool))
        sns.heatmap(df[num_cols].corr(), mask=mask, cmap='RdBu_r',
                    center=0, ax=ax, linewidths=0.3,
                    cbar_kws={'label': 'Corrélation'})
        ax.set_title("Matrice de corrélation", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with st.expander("📋 Voir les données brutes"):
        st.dataframe(df.head(100), use_container_width=True)

# ============================================================
# PAGE — PERFORMANCES
# ============================================================
elif page == "📈 Performances du modèle":
    st.header("Évaluation des Performances du Modèle")

    if model is None or df is None:
        st.error("❌ Modèle ou données non disponibles.")
        st.stop()

    if 'Class' not in df.columns:
        st.warning("⚠️ Le dataset ne contient pas de colonne 'Class'.")
        st.stop()

    try:
        from sklearn.model_selection import train_test_split
        X = df.drop('Class', axis=1).copy()
        y = df['Class']

        if scaler_amount is not None and 'Amount' in X.columns:
            X['Amount'] = scaler_amount.transform(X[['Amount']])
        if scaler_time is not None and 'Time' in X.columns:
            X['Time'] = scaler_time.transform(X[['Time']])

        cols = [c for c in feature_cols if c in X.columns] if feature_cols else X.columns.tolist()
        X = X[cols]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy",  f"{model.score(X_test, y_test):.4f}")
        col2.metric("F1-Score",  f"{f1_score(y_test, y_pred):.4f}")
        col3.metric("AUC-ROC",   f"{roc_auc_score(y_test, y_proba):.4f}")
        col4.metric("Précision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        col5.metric("Rappel",    f"{recall_score(y_test, y_pred):.4f}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Matrice de confusion")
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Normale', 'Fraude'],
                        yticklabels=['Normale', 'Fraude'],
                        linewidths=1, annot_kws={'size': 16, 'weight': 'bold'})
            ax.set_xlabel("Prédit", fontweight='bold')
            ax.set_ylabel("Réel",   fontweight='bold')
            ax.set_title("Matrice de Confusion", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            vp, fn, fp = cm[1,1], cm[1,0], cm[0,1]
            st.markdown(f"- ✅ Fraudes correctement détectées : **{vp}**")
            st.markdown(f"- ❌ Fraudes manquées (Faux Négatifs) : **{fn}** ⚠️")
            st.markdown(f"- ⚠️ Fausses alarmes (Faux Positifs) : **{fp}**")

        with c2:
            st.subheader("Courbe ROC")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color='#e74c3c', lw=2.5,
                    label=f'Random Forest (AUC={auc(fpr,tpr):.4f})')
            ax.plot([0,1],[0,1], 'k--', lw=1.5, label='Aléatoire')
            ax.fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("Courbe ROC", fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with st.expander("📋 Rapport de classification complet"):
            st.code(classification_report(
                y_test, y_pred, target_names=['Normale', 'Fraude']
            ), language='text')

    except Exception as e:
        st.error(f"❌ Erreur : {e}")

# ============================================================
# PAGE — PRÉDICTION PAR FICHIER (UNIVERSELLE)
# ============================================================
elif page == "📂 Prédiction par fichier":
    st.header("Prédiction en masse — Import CSV universel")

    if model is None:
        st.error("❌ Modèle non disponible.")
        st.stop()

    st.info("""
    📁 **Import universel** : importez n'importe quel fichier CSV.
    L'application détecte automatiquement les colonnes et s'adapte.
    """)

    uploaded = st.file_uploader("Choisissez un fichier CSV", type=['csv'])

    if uploaded:
        # ── Lecture universelle ──────────────────────────
        try:
            try:
                df_up = pd.read_csv(uploaded, sep=None, engine='python')
            except Exception:
                uploaded.seek(0)
                df_up = pd.read_csv(uploaded)

            st.success(f"✅ Fichier lu : **{len(df_up):,} lignes** — **{df_up.shape[1]} colonnes**")
            st.dataframe(df_up.head(5), use_container_width=True)
            st.markdown(f"**Colonnes détectées :** `{'` | `'.join(df_up.columns.tolist())}`")
            st.markdown("---")

        except Exception as e:
            st.error(f"❌ Impossible de lire le fichier : {e}")
            st.info("Vérifiez que le fichier est bien au format CSV (séparateur , ou ;).")
            st.stop()

        # ── Détection automatique du mode ───────────────
        cols_model   = feature_cols if feature_cols else []
        cols_present = [c for c in cols_model if c in df_up.columns]
        cols_missing = [c for c in cols_model if c not in df_up.columns]

        if cols_model and len(cols_missing) == 0:
            st.success("✅ Toutes les colonnes du modèle sont présentes — prédiction automatique !")
            mode = "direct"

        elif cols_model and len(cols_present) > 0:
            st.warning(f"⚠️ Colonnes manquantes : `{'`, `'.join(cols_missing)}` → remplacées par 0")
            mode = "partiel"

        else:
            st.info("ℹ️ Les colonnes ne correspondent pas directement — associez-les manuellement ci-dessous.")
            mode = "manuel"

        # ── Mapping manuel si nécessaire ─────────────────
        col_mapping = {}
        if mode == "manuel" and cols_model:
            st.subheader("🔧 Association des colonnes")
            st.write("Choisissez quelle colonne de votre fichier correspond à chaque variable du modèle :")
            num_up  = ["-- Ignorer (mettre 0) --"] + \
                      df_up.select_dtypes(include=[np.number]).columns.tolist()
            grid = st.columns(3)
            for i, feat in enumerate(cols_model):
                with grid[i % 3]:
                    sel = st.selectbox(f"{feat}", num_up, key=f"map_{feat}")
                    col_mapping[feat] = sel

        # ── Bouton prédiction ─────────────────────────────
        if st.button("🔮 Lancer les prédictions", type="primary", use_container_width=True):
            try:
                # Construction X selon le mode
                if mode == "direct":
                    X_up = df_up[cols_model].copy()

                elif mode == "partiel":
                    X_up = pd.DataFrame(0.0, index=df_up.index, columns=cols_model)
                    for c in cols_present:
                        X_up[c] = pd.to_numeric(df_up[c], errors='coerce').fillna(0).values

                elif mode == "manuel" and cols_model:
                    X_up = pd.DataFrame(0.0, index=df_up.index, columns=cols_model)
                    for feat, src in col_mapping.items():
                        if src != "-- Ignorer (mettre 0) --" and src in df_up.columns:
                            X_up[feat] = pd.to_numeric(df_up[src], errors='coerce').fillna(0).values

                else:
                    # Aucune info de colonnes → utiliser toutes les numériques
                    num_df = df_up.select_dtypes(include=[np.number])
                    X_up   = num_df.fillna(0)
                    st.warning("Aucune information de colonnes — utilisation de toutes les variables numériques.")

                # Normalisation
                if scaler_amount is not None and 'Amount' in X_up.columns:
                    X_up['Amount'] = scaler_amount.transform(X_up[['Amount']])
                if scaler_time is not None and 'Time' in X_up.columns:
                    X_up['Time'] = scaler_time.transform(X_up[['Time']])

                X_up = X_up.fillna(0)

                # Prédiction
                preds  = model.predict(X_up)
                probas = model.predict_proba(X_up)[:, 1]

                # Résultats
                df_res = df_up.copy()
                df_res['Prédiction']          = preds
                df_res['Probabilité_Fraude']  = probas.round(4)
                df_res['Statut']              = np.where(preds == 1, '🔴 Fraude', '✅ Normale')

                n_fraud = int(preds.sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Total analysé",         f"{len(preds):,}")
                c2.metric("Fraudes détectées",      f"{n_fraud}",
                           delta=f"{n_fraud/len(preds)*100:.2f}%")
                c3.metric("Transactions normales",  f"{len(preds)-n_fraud}")

                # Graphique
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(['Normales', 'Fraudes'],
                       [len(preds)-n_fraud, n_fraud],
                       color=['#2ecc71', '#e74c3c'], edgecolor='black')
                ax.set_title("Résultats des prédictions", fontweight='bold')
                ax.set_ylabel("Nombre de transactions")
                for i, v in enumerate([len(preds)-n_fraud, n_fraud]):
                    ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.subheader("📋 Détail des prédictions (50 premières lignes)")
                st.dataframe(
                    df_res[['Statut', 'Probabilité_Fraude']].head(50),
                    use_container_width=True
                )

                # Téléchargement
                csv_out = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Télécharger tous les résultats (CSV)",
                    csv_out,
                    file_name="resultats_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )

            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction : {e}")
                st.info("Assurez-vous que les colonnes sélectionnées contiennent des valeurs numériques.")
    else:
        st.markdown("""
        #### 💡 Ce que vous pouvez importer :
        - N'importe quel fichier **CSV** (séparateur virgule ou point-virgule)
        - **Peu importe le nom du fichier**
        - **Peu importe les noms des colonnes** — l'application s'adapte :
            - ✅ Colonnes identiques au modèle → prédiction directe
            - ⚠️ Colonnes partielles → les manquantes sont mises à 0
            - 🔧 Colonnes différentes → vous associez manuellement
        """)

# ============================================================
# PAGE — À PROPOS
# ============================================================
elif page == "ℹ️ À propos":
    st.header("À propos de ce projet")
    st.markdown("""
    ### 🎓 Partie 3 — TP2 IIA | LICENCE MTQ S6 | IUSJ Cameroun 2025-2026
    Par **Stéphane C. K. TÉKOUABOU** (PhD & Ing.)

    #### 🛠️ Technologies :
    - **Python**, **Scikit-learn**, **Pandas & NumPy**
    - **Matplotlib & Seaborn**, **Streamlit**, **Pickle**

    #### 🤖 Modèle : Random Forest Classifier
    - Optimisé par **GridSearchCV** (validation croisée 5 folds)
    - Gestion du déséquilibre par **Oversampling**

    | Métrique  | Valeur |
    |-----------|--------|
    | F1-Score  | ~0.95+ |
    | AUC-ROC   | ~0.99+ |
    | Rappel    | ~0.95+ |
    | Précision | ~0.95+ |
    """)
    if df is not None:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", f"{len(df):,}")
        if 'Class' in df.columns:
            c2.metric("Fraudes", f"{df['Class'].sum()}")
        c3.metric("Variables", f"{df.shape[1]}")
    st.success("✅ Application développée avec ❤️ pour l'apprentissage du ML")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:14px;'>"
    "🔍 Fraud Detector — TP2 IIA 2025-2026 | IUSJ Cameroun | "
    "Développé avec Streamlit & Scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
