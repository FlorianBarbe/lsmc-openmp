import streamlit as st
import subprocess
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.write("""
Cette page permet de lancer la simulation complète du modèle LSMC que nous avons développée en C++.
L’objectif est de pouvoir tester différents paramètres (prix initial, volatilité, nombre de trajectoires, etc.)
et d’observer ensuite l’impact sur les trajectoires GBM, la valeur estimée de l’option et les performances.

Le principe est simple : nous choisissons les paramètres dans la barre latérale, puis nous lançons
l’exécutable C++. Celui-ci génère deux fichiers CSV :
- un fichier contenant les trajectoires simulées,
- un fichier contenant les résultats du LSMC et les temps nécessaires à l’exécution.

Cette page joue donc le rôle de point de départ du projet : c’est ici que nous produisons les données
que nous visualisons ensuite dans les autres sections. Le but est de rendre l’exécution du LSMC
plus simple, plus flexible et totalement indépendante du code C++, tout en gardant la possibilité
de modifier le nombre de trajectoires ou le niveau de précision.

Ce système nous permet également de tester facilement l’effet de charges de calcul plus lourdes.
Lorsque le nombre de trajectoires augmente fortement, le temps d’exécution peut devenir important.
C’est pour cette raison que nous avons ensuite étudié l’impact d’OpenMP pour répartir les calculs
sur plusieurs cœurs, et que nous envisageons même la possibilité ultérieure d’exécuter certaines
parties du modèle sur carte graphique.
""")

# ---- Paramètres utilisateur ----
st.sidebar.header("Paramètres d'entrée")

S0 = st.sidebar.slider("Prix initial S₀", 50.0, 200.0, 100.0)
K = st.sidebar.slider("Strike K", 50.0, 200.0, 100.0)
r = st.sidebar.slider("Taux sans risque r", 0.0, 0.1, 0.05)
sigma = st.sidebar.slider("Volatilité σ", 0.01, 0.8, 0.2)
T = st.sidebar.slider("Maturité (années)", 0.25, 5.0, 1.0)
N_steps = st.sidebar.slider("Nombre de pas temporels", 10, 200, 50)
N_paths = st.sidebar.slider("Nombre de trajectoires", 1000, 50000, 10000)

exe_path = st.sidebar.text_input(
    "Chemin vers l'exécutable lsmc.exe",
    r"C:\Users\flole\Desktop\P1RV\lsmc\x64\Debug\lsmc.exe"
)

python_dir = os.path.dirname(os.path.dirname(__file__))
csv_paths = {
    "trajectoires": os.path.join(python_dir, "trajectoires_gbm.csv"),
    "resultats": os.path.join(python_dir, "resultats_lsmc.csv"),
}

# ---- Execution ----
st.subheader("Exécution du C++")

if st.button("Lancer la simulation"):
    cmd = [
        exe_path,
        str(S0), str(K), str(r), str(sigma),
        str(T), str(N_steps), str(N_paths)
    ]

    st.info("Commande exécutée : " + " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        st.text(result.stdout)
        st.success("Simulation terminée.")
    except Exception as e:
        st.error(f"Erreur : {e}")
