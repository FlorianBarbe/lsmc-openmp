"""
Analyse 3D : surface de scalabilité OpenMP du modèle LSMC
---------------------------------------------------------
Axes :
  X = nombre de trajectoires (N_paths)
  Y = nombre de threads
  Z = speedup = T(1 thread) / T(n threads)
"""

import subprocess, re, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour les surfaces 3D

# -------------------------------
# Paramètres
# -------------------------------
N_values = [10000*(i+1) for i in range(5)]
threads_list = [1,2,3, 4]

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
print("=== Mesure de la surface scalabilité (threads × Npaths) ===")
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
# Surface 3D
# -------------------------------
X, Y = np.meshgrid(N_values, threads_list)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, speedup, cmap='plasma', edgecolor='k', alpha=0.9)

ax.set_title("Surface de scalabilité OpenMP du modèle LSMC")
ax.set_xlabel("Nombre de trajectoires (N_paths)")
ax.set_ylabel("Nombre de threads")
ax.set_zlabel("Speedup (facteur d'accélération)")
ax.view_init(elev=25, azim=135)
fig.colorbar(surf, shrink=0.6, aspect=10, label="Facteur d'accélération")

plt.tight_layout()
plt.show()
