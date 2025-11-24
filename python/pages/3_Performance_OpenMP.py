import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.write("""
Dans cette partie de l’application, nous étudions les performances de notre programme LSMC.
Le cœur du problème est simple : pour estimer le prix d’une option américaine, nous devons simuler un très grand nombre de trajectoires. 
Chaque trajectoire représente l’évolution possible du prix du sous-jacent, et il faut en produire des milliers, parfois des dizaines de milliers,
pour que le résultat soit fiable.

Ce volume de calcul pose un problème direct : tout faire de manière séquentielle prend beaucoup de temps.
Chaque trajectoire est indépendante des autres, mais sans parallélisme, le processeur les calcule une par une.
Lorsque le nombre de trajectoires augmente, le temps d’exécution explose.

Pour réduire ce temps, nous avons activé le parallélisme du processeur grâce à OpenMP.
OpenMP permet de répartir automatiquement les trajectoires entre les différents cœurs du CPU.
Au lieu de travailler avec un seul cœur, nous faisons travailler plusieurs cœurs en même temps.
Cela réduit fortement la durée des calculs, notamment pour la partie Monte-Carlo.

Cette page compare donc les temps obtenus avec différentes valeurs du nombre de threads.
Nous observons :
- le temps d’exécution total,
- l’accélération obtenue par rapport à la version séquentielle,
- et l’effet éventuel du parallélisme sur la stabilité du prix estimé.

Enfin, même si notre travail se concentre ici sur le parallélisme CPU, il serait possible d’aller plus loin.
Une autre piste consiste à faire exécuter les calculs directement par la carte graphique (GPU),
ce qui peut accélérer encore plus les simulations. Nous ne l’avons pas encore testé,
mais cela ouvre une possibilité d’évolution du projet.
""")

python_dir = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(python_dir, "resultats_lsmc.csv")

# -----------------------------
# 1) Charger le CSV
# -----------------------------
if not os.path.exists(csv_path):
    st.warning("Aucun fichier resultats_lsmc.csv trouvé.")
    st.stop()

df = pd.read_csv(csv_path)

st.subheader("Données brutes")
st.dataframe(df, use_container_width=True)

# -----------------------------
# 2) Recalculer le speedup
# -----------------------------
# Ligne seq = Threads == 1
t_seq = df[df["Threads"] == 1]["Temps"].iloc[-1]

df["Speedup"] = t_seq / df["Temps"]

# -----------------------------
# 3) Graphique : temps vs threads
# -----------------------------
st.subheader("Temps d'exécution selon le nombre de threads")

fig1, ax1 = plt.subplots()
ax1.plot(df["Threads"], df["Temps"], marker="o")
ax1.set_xlabel("Threads")
ax1.set_ylabel("Temps (s)")
ax1.set_xticks(df["Threads"])
ax1.set_title("Temps d'exécution (séquentiel vs OpenMP)")

st.pyplot(fig1)

# -----------------------------
# 4) Graphique : speedup
# -----------------------------
st.subheader("Speedup (T_seq / T_par)")

fig2, ax2 = plt.subplots()
ax2.plot(df["Threads"], df["Speedup"], marker="s", color="green")
ax2.axhline(1, color="red", linestyle="--")
ax2.set_xlabel("Threads")
ax2.set_ylabel("Speedup")
ax2.set_xticks(df["Threads"])
ax2.set_title("Accélération OpenMP")

st.pyplot(fig2)

# -----------------------------
# 5) Prix vs threads
# -----------------------------
st.subheader("Prix estimé selon le nombre de threads")

fig3, ax3 = plt.subplots()
ax3.plot(df["Threads"], df["Prix"], marker="o", color="orange")
ax3.set_xlabel("Threads")
ax3.set_ylabel("Prix estimé")
ax3.set_xticks(df["Threads"])
ax3.set_title("Impact du parallélisme sur le prix")

st.pyplot(fig3)
