# ================================================================
# UI.py — Interface Streamlit du projet LSMC (C++ + OpenMP)
# ================================================================
# - Interface de contrôle des paramètres de simulation
# - Exécution du binaire C++ avec ces paramètres
# - Lecture automatique des CSV produits
# - Visualisation des trajectoires et des performances
# ================================================================

import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="LSMC OpenMP", layout="wide")
st.title("💻 Simulation LSMC (C++ + OpenMP)")

# ------------------------------------------------
# === 1. Paramètres utilisateur ===
# ------------------------------------------------
st.sidebar.header("Paramètres d'entrée")

S0 = st.sidebar.slider("Prix initial S₀", 50.0, 200.0, 100.0)
K = st.sidebar.slider("Strike K", 50.0, 200.0, 100.0)
r = st.sidebar.slider("Taux sans risque r", 0.0, 0.1, 0.05)
sigma = st.sidebar.slider("Volatilité σ", 0.01, 0.5, 0.2)
T = st.sidebar.slider("Maturité (années)", 0.25, 5.0, 1.0)
N_steps = st.sidebar.slider("Nombre de pas temporels", 10, 200, 50)
N_paths = st.sidebar.slider("Nombre de trajectoires", 1000, 50000, 10000)
exe_path = st.sidebar.text_input(
    "Chemin vers l'exécutable lsmc.exe",
    r"C:\Users\flole\Desktop\lsmc\x64\Debug\lsmc.exe"
)

output_dir = "../../output"
csv_paths = {
    "trajectoires": os.path.join(output_dir, "trajectoires_gbm.csv"),
    "resultats": os.path.join(output_dir, "resultats_lsmc.csv"),
}

# ------------------------------------------------
# === 2. Lancement du C++ ===
# ------------------------------------------------
st.subheader("⚙️ Exécution du modèle C++")

if st.button("Lancer la simulation"):
    cmd = [
        exe_path,
        str(S0), str(K), str(r), str(sigma),
        str(T), str(N_steps), str(N_paths)
    ]
    st.info(f"Commande exécutée : {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        st.text(result.stdout)
        st.success("✅ Simulation terminée.")
    except subprocess.CalledProcessError as e:
        st.error("Erreur lors de l'exécution du modèle C++")
        st.text(e.stderr)
    except FileNotFoundError:
        st.error("❌ Fichier lsmc.exe introuvable. Vérifie le chemin indiqué.")
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")

# ------------------------------------------------
# === 3. Lecture des trajectoires simulées ===
# ------------------------------------------------
st.subheader("📈 Trajectoires simulées")

if os.path.exists(csv_paths["trajectoires"]):
    try:
        paths = np.loadtxt(csv_paths["trajectoires"], delimiter=",")
        t = np.linspace(0, T, paths.shape[1])

        fig, ax = plt.subplots(figsize=(8, 4))
        for p in paths[:50]:  # affiche les 50 premières trajectoires
            ax.plot(t, p, lw=0.8, alpha=0.5, color="gray")
        ax.plot(t, np.mean(paths, axis=0), lw=2, color="cyan", label="Moyenne")
        ax.set_xlabel("Temps (années)")
        ax.set_ylabel("Prix du sous-jacent")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Impossible de lire {csv_paths['trajectoires']} : {e}")
else:
    st.info("Aucune trajectoire simulée encore disponible.")

# ------------------------------------------------
# === 4. Lecture des résultats agrégés ===
# ------------------------------------------------
st.subheader("📊 Analyse des performances (resultats_lsmc.csv)")

if os.path.exists(csv_paths["resultats"]):
    try:
        df = pd.read_csv(csv_paths["resultats"], encoding="latin1")
        st.dataframe(df.tail(10), use_container_width=True)

        # Courbe 1 : temps séquentiel vs OpenMP
        fig1, ax1 = plt.subplots()
        ax1.plot(df["N_paths"], df["Temps_Seq"], marker="o", label="Séquentiel")
        ax1.plot(df["N_paths"], df["Temps_OpenMP"], marker="o", label="OpenMP")
        ax1.set_xlabel("N_paths")
        ax1.set_ylabel("Temps (s)")
        ax1.set_title("Comparaison des temps d'exécution")
        ax1.legend()
        st.pyplot(fig1)

        # Courbe 2 : Speedup
        fig2, ax2 = plt.subplots()
        ax2.plot(df["N_paths"], df["Speedup"], color="green", marker="s")
        ax2.axhline(1, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("N_paths")
        ax2.set_ylabel("Speedup (T_seq / T_par)")
        ax2.set_title("Accélération OpenMP")
        st.pyplot(fig2)

        # Courbe 3 : Convergence du prix
        fig3, ax3 = plt.subplots()
        ax3.plot(df["N_paths"], df["Prix_Seq"], label="Séquentiel", linestyle="--")
        ax3.plot(df["N_paths"], df["Prix_OpenMP"], label="OpenMP", linestyle="-")
        ax3.set_xlabel("N_paths")
        ax3.set_ylabel("Prix estimé")
        ax3.set_title("Convergence du prix estimé")
        ax3.legend()
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"Erreur de lecture du CSV : {e}")
else:
    st.info("Le fichier resultats_lsmc.csv n'a pas encore été généré.")
