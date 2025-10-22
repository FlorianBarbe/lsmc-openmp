import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.subheader("📊 Analyse des performances OpenMP")

# --- Lecture du CSV ---
try:
    df = pd.read_csv(r"C:\Users\flole\Desktop\lsmc\x64\Debug\resultats_lsmc.csv", encoding="latin1")  # éviter bug d’encodage
    st.dataframe(df.head())

    # --- Courbe 1 : Temps séquentiel vs parallèle ---
    fig1, ax1 = plt.subplots()
    ax1.plot(df["N_paths"], df["Temps_Séqu"], label="Temps séquentiel", marker="o")
    ax1.plot(df["N_paths"], df["Temps_OpenMP"], label="Temps OpenMP", marker="o")
    ax1.set_xlabel("Nombre de trajectoires (N_paths)")
    ax1.set_ylabel("Temps (secondes)")
    ax1.set_title("Comparaison du temps d’exécution")
    ax1.legend()
    st.pyplot(fig1)

    # --- Courbe 2 : Speedup ---
    fig2, ax2 = plt.subplots()
    ax2.plot(df["N_paths"], df["Speedup"], color="green", marker="s")
    ax2.set_xlabel("Nombre de trajectoires (N_paths)")
    ax2.set_ylabel("Speedup (T_seq / T_par)")
    ax2.set_title("Accélération obtenue grâce à OpenMP")
    st.pyplot(fig2)

    # --- Courbe 3 : Convergence du prix estimé ---
    fig3, ax3 = plt.subplots()
    ax3.plot(df["N_paths"], df["Prix_Séqu"], label="Séquentiel", linestyle="--")
    ax3.plot(df["N_paths"], df["Prix_OpenMP"], label="OpenMP", linestyle="-")
    ax3.set_xlabel("Nombre de trajectoires (N_paths)")
    ax3.set_ylabel("Prix estimé de l’option")
    ax3.set_title("Convergence du prix estimé")
    ax3.legend()
    st.pyplot(fig3)

except FileNotFoundError:
    st.error("⚠️ Fichier resultats_lsmc.csv introuvable. Vérifie son emplacement.")
except Exception as e:
    st.error(f"Erreur de lecture du CSV : {e}")
