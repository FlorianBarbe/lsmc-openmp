import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

st.title("📈 Trajectoires GBM")

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
    ax.set_title("Trajectoires GBM simulées")
    ax.legend()

    st.pyplot(fig)

else:
    st.warning("Aucune trajectoire simulée pour l’instant.")
