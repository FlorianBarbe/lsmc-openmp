# ================================================================
# UI.py — Page d'accueil de l'application multipages LSMC
# ================================================================

import streamlit as st

st.set_page_config(page_title="LSMC OpenMP", layout="wide")

# Titre principal
st.title("📘 Projet LSMC — Pricing d’options américaines (C++ + OpenMP)")

# Introduction
st.write("""
Bienvenue dans l'application interactive du projet **Least Squares Monte Carlo (LSMC)** 
développée en C++ avec parallélisation OpenMP.

Cette interface Streamlit permet de :

### 🔧 1. Lancer une simulation complète
- génération de trajectoires GBM,
- calcul backward LSMC (régression OLS),
- exécution séquentielle + OpenMP,
- export automatique des CSV.

### 📈 2. Visualiser les trajectoires simulées
- jusqu'à 50 trajectoires affichées,
- moyenne analytique,
- comparaison variance / volatilité.

### 🚀 3. Analyser les performances du code C++
- comparaison séquentiel vs OpenMP,
- speedup,
- influence de N_paths et N_steps.

### 📚 4. Comprendre la méthode LSMC
- rappel du modèle GBM,
- régression polynomiale (Longstaff & Schwartz),
- backward induction,
- structure de ton code C++.

Utilisez le menu de gauche pour accéder aux différentes pages.
""")

st.info("Sélectionnez une page dans la barre latérale pour commencer.")
