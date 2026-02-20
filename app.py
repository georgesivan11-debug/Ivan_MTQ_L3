import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    f1_score, roc_auc_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Détection de Fraude — IUSJ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PERSONNALISÉ
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin: 0.3rem;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .fraud-alert {
        background: #FDECEA;
        border-left: 5px solid #e74c3c;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: #c0392b;
    }
    .normal-alert {
        background: #EAFAF1;
        border-left: 5px solid #2ecc71;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e8449;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1F4E79;
        border-bottom: 3px solid #2E75B6;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f4f8;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# ============================================================
@st.cache_resource
def charger_modele():
    with open('creditcard.pkl', 'rb') as f:
        modele = pickle.load(f)
    return modele

@st.cache_data
def charger_donnees():
    df = pd.read_csv('creditcard.csv')
    return df

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

try:
    modele = charger_modele()
    df     = charger_donnees()
    X_train, X_test, y_train, y_test, scaler = preparer_donnees(df)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"❌ Erreur de chargement : {e}")
    st.stop()

# ============================================================
# EN-TÊTE
# ============================================================
st.markdown('<div class="main-title">🔍 Détection de Fraude par Carte de Crédit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Partie 3 — TP2 IIA S6 | Institut Universitaire Saint Jean du Cameroun | Année 2025-2026</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/260px-Scikit_learn_logo_small.svg.png", width=120)
    st.markdown("## ⚙️ Informations")
    st.info(f"""
    **Modèle :** Random Forest  
    **Algorithme :** Ensemble (Bagging)  
    **Dataset :** Credit Card Fraud  
    **Instances :** {len(df):,}  
    **Features :** {df.shape[1]-1}  
    **Fraudes :** {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)
    """)
    st.markdown("---")
    st.markdown("### 📚 Bibliothèques utilisées")
    libs = ["scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "streamlit", "pickle"]
    for lib in libs:
        st.markdown(f"• `{lib}`")
    st.markdown("---")
    st.caption("© 2025-2026 — IUSJ Cameroun")

# ============================================================
# ONGLETS PRINCIPAUX
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploration des données",
    "🤖 Prédiction individuelle",
    "📈 Performances du modèle",
    "📂 Prédiction par fichier"
])

# ══════════════════════════════════════════════════════════
# ONGLET 1 — EXPLORATION
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📊 Exploration et Visualisation des Données</div>', unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h2>{len(df):,}</h2><p>Transactions totales</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h2>{df["Class"].sum()}</h2><p>Fraudes détectées</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h2>{df["Class"].mean()*100:.2f}%</h2><p>Taux de fraude</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h2>{df.shape[1]-1}</h2><p>Variables explicatives</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    col_g, col_d = st.columns(2)

    with col_g:
        st.subheader("Répartition des classes")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df['Class'].value_counts()
        bars = ax.bar(['Normale (0)', 'Fraude (1)'], counts.values,
                      color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{val:,}\n({val/len(df)*100:.2f}%)', ha='center', fontweight='bold', fontsize=11)
        ax.set_ylabel("Nombre de transactions")
        ax.set_title("Distribution des classes", fontweight='bold')
        ax.set_ylim(0, counts.max() * 1.2)
        st.pyplot(fig)
        plt.close()

    with col_d:
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
        st.pyplot(fig)
        plt.close()

    st.subheader("Distribution des features V1–V10 par classe")
    features_v = [f'V{i}' for i in range(1, 11)]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()
    for i, feat in enumerate(features_v):
        data = [df[df['Class']==0][feat].values, df[df['Class']==1][feat].values]
        bp = axes[i].boxplot(data, patch_artist=True, medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        axes[i].set_title(feat, fontweight='bold')
        axes[i].set_xticklabels(['Normal', 'Fraude'], fontsize=8)
    plt.suptitle("Distribution des features V1–V10 par classe", fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, ax=ax,
                linewidths=0.3, cbar_kws={'label': 'Corrélation'})
    ax.set_title("Matrice de corrélation", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 Voir les données brutes"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Affichage des 100 premières lignes sur {len(df):,}")

# ══════════════════════════════════════════════════════════
# ONGLET 2 — PRÉDICTION INDIVIDUELLE
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🤖 Prédiction sur une Transaction</div>', unsafe_allow_html=True)
    st.info("💡 Entrez manuellement les caractéristiques d'une transaction pour obtenir une prédiction en temps réel.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Paramètres de la transaction")
        amount = st.number_input("💰 Montant de la transaction (€)", min_value=0.0, max_value=25000.0, value=150.0, step=10.0)
        time   = st.number_input("⏱️ Temps depuis la 1ère transaction (secondes)", min_value=0.0, value=50000.0, step=1000.0)

        st.markdown("**Variables PCA (V1 à V10) :**")
        cols = st.columns(2)
        v_vals = {}
        for i in range(1, 11):
            col_idx = (i-1) % 2
            with cols[col_idx]:
                v_vals[f'V{i}'] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"v{i}", format="%.3f")

        st.markdown("---")
        utiliser_exemple = st.toggle("🎲 Utiliser un exemple aléatoire du dataset")
        if utiliser_exemple:
            exemple_classe = st.radio("Type d'exemple", ["Normale", "Fraude"], horizontal=True)
            classe_val = 0 if exemple_classe == "Normale" else 1
            exemple = df[df['Class'] == classe_val].sample(1, random_state=np.random.randint(0, 100))
            amount = float(exemple['Amount'].values[0])
            time   = float(exemple['Time'].values[0])
            for i in range(1, 11):
                v_vals[f'V{i}'] = float(exemple[f'V{i}'].values[0])
            st.success(f"Exemple chargé ({exemple_classe})")

    with col_right:
        st.subheader("Résultat de la prédiction")

        if st.button("🔍 Analyser cette transaction", type="primary", use_container_width=True):
            # Construction du vecteur de features
            input_data = {'Time': time, 'Amount': amount}
            for i in range(1, 11):
                input_data[f'V{i}'] = v_vals[f'V{i}']
            input_df = pd.DataFrame([input_data])[X_test.columns]

            # Normalisation
            input_df['Amount'] = scaler.transform(input_df[['Amount']])
            input_df['Time']   = scaler.transform(input_df[['Time']])

            # Prédiction
            prediction = modele.predict(input_df)[0]
            proba      = modele.predict_proba(input_df)[0]

            # Affichage résultat
            if prediction == 1:
                st.markdown('<div class="fraud-alert">🚨 TRANSACTION FRAUDULEUSE DÉTECTÉE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="normal-alert">✅ TRANSACTION LÉGITIME</div>', unsafe_allow_html=True)

            st.markdown("---")

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Probabilité — Normale", f"{proba[0]*100:.1f}%")
            col_m2.metric("Probabilité — Fraude",  f"{proba[1]*100:.1f}%")

            # Jauge de risque
            st.markdown("**Niveau de risque :**")
            risque = proba[1]
            couleur = "#e74c3c" if risque > 0.5 else ("#f39c12" if risque > 0.2 else "#2ecc71")
            st.progress(risque)
            st.markdown(f"<p style='color:{couleur}; font-weight:bold; font-size:1.1rem;'>Risque de fraude : {risque*100:.1f}%</p>", unsafe_allow_html=True)

            # Résumé de la transaction
            st.markdown("---")
            st.markdown("**Récapitulatif :**")
            st.markdown(f"- Montant : **{amount:.2f} €**")
            st.markdown(f"- Décision : **{'🔴 FRAUDE' if prediction == 1 else '🟢 NORMALE'}**")
            st.markdown(f"- Confiance du modèle : **{max(proba)*100:.1f}%**")

# ══════════════════════════════════════════════════════════
# ONGLET 3 — PERFORMANCES
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📈 Évaluation des Performances du Modèle</div>', unsafe_allow_html=True)

    y_pred  = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]

    # Métriques principales
    f1    = f1_score(y_test, y_pred)
    auc_s = roc_auc_score(y_test, y_proba)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)
    acc   = modele.score(X_test, y_test)

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, label, val in zip(
        [col1, col2, col3, col4, col5],
        ["Accuracy", "F1-Score", "AUC-ROC", "Précision", "Rappel"],
        [acc, f1, auc_s, prec, rec]
    ):
        col.markdown(f'<div class="metric-card"><h2>{val:.4f}</h2><p>{label}</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    col_g, col_d = st.columns(2)

    with col_g:
        st.subheader("Matrice de confusion")
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normale', 'Fraude'],
                    yticklabels=['Normale', 'Fraude'],
                    linewidths=1, linecolor='gray',
                    annot_kws={'size': 16, 'weight': 'bold'})
        ax.set_xlabel("Prédit", fontweight='bold')
        ax.set_ylabel("Réel", fontweight='bold')
        ax.set_title("Matrice de Confusion", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        vp = cm[1,1]; fn = cm[1,0]; fp = cm[0,1]
        st.markdown(f"""
        - ✅ **Vrais Positifs (Fraudes correctes)** : {vp}
        - ❌ **Faux Négatifs (Fraudes manquées)** : {fn} ⚠️
        - ⚠️ **Faux Positifs (Fausses alarmes)** : {fp}
        """)

    with col_d:
        st.subheader("Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'Random Forest (AUC = {auc_val:.4f})')
        ax.plot([0,1],[0,1], 'k--', lw=1.5, label='Aléatoire (AUC = 0.5)')
        ax.fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
        ax.set_xlabel("Taux de Faux Positifs (FPR)", fontweight='bold')
        ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontweight='bold')
        ax.set_title("Courbe ROC", fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Importance des variables")
    importances = modele.feature_importances_
    feat_names  = X_test.columns.tolist()
    idx = np.argsort(importances)[::-1][:12]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_imp = ['#e74c3c' if i < 3 else '#3498db' for i in range(len(idx))]
    ax.bar([feat_names[i] for i in idx], importances[idx], color=colors_imp, edgecolor='black')
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance (Gini)")
    ax.set_title("Top 12 variables les plus importantes", fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 Rapport de classification complet"):
        report = classification_report(y_test, y_pred, target_names=['Normale', 'Fraude'])
        st.code(report, language='text')

# ══════════════════════════════════════════════════════════
# ONGLET 4 — PRÉDICTION PAR FICHIER
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📂 Prédiction sur un Fichier CSV</div>', unsafe_allow_html=True)
    st.info("📁 Importez un fichier CSV contenant des transactions pour obtenir des prédictions en masse.")

    uploaded = st.file_uploader("Choisissez un fichier CSV", type=['csv'])

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.success(f"✅ Fichier chargé : {len(df_upload):,} lignes, {df_upload.shape[1]} colonnes")
            st.dataframe(df_upload.head(), use_container_width=True)

            # Prédiction si les colonnes correspondent
            required = X_test.columns.tolist()
            missing_cols = [c for c in required if c not in df_upload.columns]

            if missing_cols:
                st.error(f"❌ Colonnes manquantes : {missing_cols}")
            else:
                X_up = df_upload[required].copy()
                X_up['Amount'] = scaler.transform(X_up[['Amount']])
                X_up['Time']   = scaler.transform(X_up[['Time']])

                preds  = modele.predict(X_up)
                probas = modele.predict_proba(X_up)[:, 1]

                df_upload['Prédiction'] = preds
                df_upload['Probabilité_Fraude'] = probas.round(4)
                df_upload['Statut'] = df_upload['Prédiction'].map({0: '✅ Normale', 1: '🔴 Fraude'})

                n_fraud = preds.sum()
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total analysé", f"{len(preds):,}")
                c2.metric("Fraudes détectées", f"{n_fraud}", delta=f"{n_fraud/len(preds)*100:.2f}%")
                c3.metric("Transactions normales", f"{len(preds)-n_fraud}")

                st.dataframe(
                    df_upload[['Statut', 'Probabilité_Fraude'] + required[:5]],
                    use_container_width=True
                )

                csv_out = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Télécharger les résultats",
                    csv_out,
                    file_name="resultats_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.markdown("""
        #### Format attendu du fichier CSV :
        Le fichier doit contenir les colonnes suivantes :
        `Time`, `V1`, `V2`, ..., `V10`, `Amount`
        
        > 💡 Vous pouvez utiliser le fichier `creditcard.csv` fourni comme exemple.
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
    TP2 IIA — Implémentation et déploiement des modèles de ML | LICENCE MTQ S6 | IUSJ Cameroun 2025-2026<br>
    Modèle : <b>Random Forest Classifier</b> | Framework : <b>Streamlit</b>
</div>
""", unsafe_allow_html=True)
