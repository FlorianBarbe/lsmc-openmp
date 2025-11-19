import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.title("🚀 Performance : OpenMP vs Séquentiel")

python_dir = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(python_dir, "resultats_lsmc.csv")

# -----------------------------
# 1) Charger le CSV
# -----------------------------
if not os.path.exists(csv_path):
    st.warning("Aucun fichier resultats_lsmc.csv trouvé.")
    st.stop()

df = pd.read_csv(csv_path)

st.subheader("📄 Données brutes")
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
st.subheader("⏱️ Temps d'exécution selon le nombre de threads")

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
st.subheader("⚡ Speedup (T_seq / T_par)")

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
st.subheader("📈 Prix estimé selon le nombre de threads")

fig3, ax3 = plt.subplots()
ax3.plot(df["Threads"], df["Prix"], marker="o", color="orange")
ax3.set_xlabel("Threads")
ax3.set_ylabel("Prix estimé")
ax3.set_xticks(df["Threads"])
ax3.set_title("Impact du parallélisme sur le prix")

st.pyplot(fig3)
