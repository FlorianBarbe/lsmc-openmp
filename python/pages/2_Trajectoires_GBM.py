import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

st.title("Trajectoires GBM simulées")

st.write("""
Cette page permet de visualiser les trajectoires du mouvement brownien géométrique (GBM)
que nous générons en C++ lors de la simulation.  
Le fichier CSV contient un ensemble de trajectoires simulées à partir du modèle :

    dS_t = μ S_t dt + σ S_t dW_t

Nous affichons ici un sous-ensemble des trajectoires pour observer la dispersion, ainsi
que leur moyenne, qui suit l’évolution analytique attendue du modèle.  
Cette visualisation nous permet de vérifier la cohérence de la simulation, de contrôler
l’effet de la volatilité et du nombre de pas de temps, et de disposer d’un support clair
avant l’étape de régression LSMC.
""")

python_dir = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(python_dir, "trajectoires_gbm.csv")

if os.path.exists(csv_path):
    paths = np.loadtxt(csv_path, delimiter=",")
    t = np.linspace(0, 1, paths.shape[1])

    fig, ax = plt.subplots(figsize=(10,4))

    for p in paths[:50]:
        ax.plot(t, p, color="gray", alpha=0.5)

    ax.plot(t, paths.mean(axis=0), lw=3, color="cyan", label="Moyenne")

    ax.set_xlabel("Temps (années)")
    ax.set_ylabel("Prix du sous-jacent")
    ax.set_title("Visualisation des trajectoires GBM")
    ax.legend()

    st.pyplot(fig)

else:
    st.warning("Aucune trajectoire simulée pour le moment.")
