"""
Heatmap de scalabilité OpenMP du modèle LSMC
---------------------------------------------
Affiche le speedup (facteur d'accélération) en fonction de :
  - X : nombre de trajectoires (N_paths)
  - Y : nombre de threads
"""

import subprocess, re, os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Paramètres
# -------------------------------
N_values = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
threads_list = [1, 2, 4, 8, 16]

times = np.zeros((len(threads_list), len(N_values)))

# -------------------------------
# Fonction d'exécution
# -------------------------------
def run_lsmc(n_paths, n_threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)
    result = subprocess.run(["lsmc.exe", str(n_paths)], capture_output=True, text=True, env=env)
    match_time = re.search(r"Temps d'exécution\s*:\s*([0-9.]+)", result.stdout)
    t = float(match_time.group(1)) if match_time else np.nan
    return t

# -------------------------------
# Mesure des temps
# -------------------------------
print("=== Heatmap scalabilité (threads × Npaths) ===")
for i, nt in enumerate(threads_list):
    for j, n in enumerate(N_values):
        print(f"{nt} threads, {n} trajectoires...")
        times[i, j] = run_lsmc(n, nt)

# -------------------------------
# Calcul du speedup
# -------------------------------
t1 = times[0, :]  # temps à 1 thread
speedup = np.zeros_like(times)
for i in range(len(threads_list)):
    speedup[i, :] = t1 / times[i, :]

# -------------------------------
# Heatmap
# -------------------------------
plt.figure(figsize=(9, 6))
plt.imshow(speedup, aspect='auto', origin='lower', cmap='plasma',
           extent=[min(N_values), max(N_values), min(threads_list), max(threads_list)])
plt.colorbar(label="Speedup (facteur d'accélération)")

plt.title("Heatmap de scalabilité OpenMP du modèle LSMC")
plt.xlabel("Nombre de trajectoires (N_paths)")
plt.ylabel("Nombre de threads")

# Lignes de contour (optionnel pour lisibilité)
CS = plt.contour(speedup, levels=np.linspace(np.nanmin(speedup), np.nanmax(speedup), 6),
                 colors='k', linewidths=0.5, origin='lower',
                 extent=[min(N_values), max(N_values), min(threads_list), max(threads_list)])
plt.clabel(CS, inline=True, fontsize=8, fmt="%.2f")

plt.tight_layout()
plt.show()
