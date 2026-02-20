# ============================================================
# TP2 IIA — Partie 3 : Détection de Fraude par Carte de Crédit
# Script d'entraînement complet
# LICENCE MTQ S6 — IUSJ Cameroun 2025-2026
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("  DÉTECTION DE FRAUDE — SCRIPT D'ENTRAÎNEMENT COMPLET")
print("="*60)

# ============================================================
# 1. CHARGEMENT ET DESCRIPTION DES DONNÉES
# ============================================================
print("\n📦 Chargement des données...")
df = pd.read_csv('creditcard.csv')

print(f"\n  Dimensions       : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"  Transactions normales  : {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.2f}%)")
print(f"  Transactions fraudules : {(df['Class']==1).sum():,} ({(df['Class']==1).mean()*100:.2f}%)")
print(f"  Valeurs manquantes     : {df.isnull().sum().sum()}")
print("\nStatistiques descriptives :")
print(df.describe().round(3))

# ============================================================
# 2. VISUALISATION
# ============================================================
print("\n📊 Génération des visualisations...")

# Distribution des classes
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df['Class'].value_counts()
axes[0].bar(['Normale (0)', 'Fraude (1)'], counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[0].set_title('Distribution des classes', fontweight='bold')
axes[0].set_ylabel('Nombre')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, f'{v:,}\n({v/len(df)*100:.2f}%)', ha='center', fontweight='bold')
axes[1].pie(counts.values, labels=['Normale', 'Fraude'],
            colors=['#2ecc71', '#e74c3c'], autopct='%1.2f%%',
            startangle=90, explode=(0, 0.1))
axes[1].set_title('Proportion des classes', fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_distribution_classes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig1_distribution_classes.png")

# Boxplots V1-V10
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
plt.savefig('fig2_boxplots_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig2_boxplots_features.png")

# ============================================================
# 3. PRÉPARATION DES DONNÉES
# ============================================================
print("\n⚙️  Préparation des données...")
X = df.drop('Class', axis=1).copy()
y = df['Class']

scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time']   = scaler.fit_transform(X[['Time']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Oversampling
X_train_df = X_train.copy()
X_train_df['Class'] = y_train.values
df_maj = X_train_df[X_train_df['Class']==0]
df_min = X_train_df[X_train_df['Class']==1]
df_min_up = resample(df_min, replace=True, n_samples=len(df_maj)//3, random_state=42)
df_bal = pd.concat([df_maj, df_min_up])
X_train_bal = df_bal.drop('Class', axis=1)
y_train_bal = df_bal['Class']

print(f"  Train : {X_train_bal.shape[0]:,} instances | Test : {X_test.shape[0]:,} instances")
print(f"  Fraudes train (après oversampling) : {y_train_bal.sum()}")

# ============================================================
# 4. CLASSIFIEUR DE RÉFÉRENCE (BASELINE)
# ============================================================
print("\n📏 Classifieur constant (baseline)...")
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train_bal, y_train_bal)
print(f"  Accuracy  : {dummy.score(X_test, y_test):.4f}")
print(f"  F1-Score  : {f1_score(y_test, dummy.predict(X_test)):.4f}")
print("  ⚠️ Le classifieur constant prédit TOUJOURS la classe majoritaire.")

# ============================================================
# 5. ENTRAÎNEMENT ET COMPARAISON DES MODÈLES
# ============================================================
print("\n🤖 Entraînement des modèles...")

models = {
    'KNN (k=7)'           : KNeighborsClassifier(n_neighbors=7),
    'Arbre de Décision'   : DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
    'Régression Logistique': LogisticRegression(random_state=42, max_iter=500),
}

results = {}
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'Accuracy'  : model.score(X_test, y_test),
        'F1'        : f1_score(y_test, y_pred),
        'Précision' : precision_score(y_test, y_pred, zero_division=0),
        'Rappel'    : recall_score(y_test, y_pred),
        'AUC-ROC'   : roc_auc_score(y_test, y_proba),
    }
    print(f"  {name:<25} | F1: {results[name]['F1']:.4f} | AUC: {results[name]['AUC-ROC']:.4f}")

df_results = pd.DataFrame(results).T.sort_values('F1', ascending=False)
print("\n📊 Tableau comparatif :")
print(df_results.round(4).to_string())

# ============================================================
# 6. INFLUENCE DU PARAMÈTRE K (KNN)
# ============================================================
print("\n🔬 Étude de l'influence du paramètre k...")
k_values = range(1, 31, 2)
f1_scores = []
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_bal, y_train_bal)
    f1_scores.append(f1_score(y_test, knn_k.predict(X_test)))

best_k = list(k_values)[np.argmax(f1_scores)]
print(f"  Meilleur k : {best_k} (F1 = {max(f1_scores):.4f})")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, f1_scores, 'g-D', markersize=5)
ax.axvline(x=best_k, color='red', linestyle='--', label=f'Meilleur k={best_k}')
ax.set_xlabel('Valeur de k')
ax.set_ylabel('F1-Score')
ax.set_title('Influence de k sur le F1-Score', fontweight='bold')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('fig3_influence_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig3_influence_k.png")

# ============================================================
# 7. OPTIMISATION PAR GRIDSEARCHCV (RANDOM FOREST)
# ============================================================
print("\n🎯 GridSearchCV sur Random Forest (patience ~1 min)...")
param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_features' : ['sqrt', 'log2'],
    'max_depth'    : [5, 10, 15, None]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0
)
grid_rf.fit(X_train_bal, y_train_bal)
rf_best = grid_rf.best_estimator_

print(f"  Meilleurs paramètres : {grid_rf.best_params_}")
print(f"  F1 CV moyen  : {grid_rf.best_score_:.4f}")
print(f"  F1 test      : {f1_score(y_test, rf_best.predict(X_test)):.4f}")
print(f"  AUC test     : {roc_auc_score(y_test, rf_best.predict_proba(X_test)[:,1]):.4f}")

# ============================================================
# 8. COURBES ROC
# ============================================================
print("\n📈 Génération des courbes ROC...")
top_models = {
    'Random Forest Optimisé' : rf_best,
    'Gradient Boosting'      : models['Gradient Boosting'],
    'KNN (k=7)'              : models['KNN (k=7)'],
}
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = ['#e74c3c', '#3498db', '#2ecc71']
for (nom, mdl), color in zip(top_models.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, mdl.predict_proba(X_test)[:,1])
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{nom} (AUC={auc(fpr,tpr):.4f})')
ax.plot([0,1],[0,1], 'k--', lw=1.5, label='Aléatoire (AUC=0.5)')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('Courbes ROC — Top 3 modèles', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_courbes_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig4_courbes_roc.png")

# ============================================================
# 9. IMPORTANCE DES VARIABLES
# ============================================================
print("\n🔑 Importance des variables...")
importances = rf_best.feature_importances_
feat_names  = X_test.columns.tolist()
idx = np.argsort(importances)[::-1][:12]
top3 = [feat_names[i] for i in idx[:3]]
print(f"  Top 3 variables (RF optimisé) : {top3}")

fig, ax = plt.subplots(figsize=(12, 5))
colors_imp = ['#e74c3c' if i < 3 else '#3498db' for i in range(len(idx))]
ax.bar([feat_names[i] for i in idx], importances[idx], color=colors_imp, edgecolor='black')
ax.set_ylabel("Importance (Gini)")
ax.set_title("Top 12 variables les plus importantes — Random Forest Optimisé", fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('fig5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ fig5_feature_importance.png")

# ============================================================
# 10. RAPPORT FINAL
# ============================================================
print("\n" + "="*60)
print("  RAPPORT FINAL — RANDOM FOREST OPTIMISÉ")
print("="*60)
y_pred_final = rf_best.predict(X_test)
print(classification_report(y_test, y_pred_final, target_names=['Normale', 'Fraude']))

# ============================================================
# 11. SAUVEGARDE DU MODÈLE
# ============================================================
print("💾 Sauvegarde du modèle...")
with open('creditcard.pkl', 'wb') as f:
    pickle.dump(rf_best, f)
print("  ✅ creditcard.pkl sauvegardé !")

print("\n" + "="*60)
print("  TP TERMINÉ AVEC SUCCÈS ! 🎉")
print("  Lancez maintenant : streamlit run streamlit_app.py")
print("="*60)
