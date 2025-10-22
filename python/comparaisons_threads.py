"""
Analyse complète du modèle LSMC (Séquentiel, OpenMP et Scalabilité réelle)
--------------------------------------------------------------------------
Ce script :
  1. Compare Séquentiel vs OpenMP (prix + temps)
  2. Calcule le speedup par N_paths
  3. Mesure la scalabilité réelle (1, 2, 4, 8, 16 threads)
  4. Trace 4 graphiques clairs
"""

import subprocess, re, os
import matplotlib.pyplot as plt

# ================================
# PARAMÈTRES
# ================================
N_values = [1000 * (i + 1) for i in range(20)]   # 1000 → 20000
threads_scaling = [1, 2, 4, 8, 16]
N_paths_scaling = 20000

data = {"sequentiel": {"price": [], "time": []},
        "openmp": {"price": [], "time": []}}

# ================================
# FONCTION D'EXÉCUTION
# ================================
def run_lsmc(n_paths, n_threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)
    result = subprocess.run(["lsmc.exe", str(n_paths)],
                            capture_output=True, text=True, env=env)
    match_price = re.search(r"Prix estimé du put américain\s*:\s*([0-9.]+)", result.stdout)
    match_time = re.search(r"Temps d'exécution\s*:\s*([0-9.]+)", result.stdout)
    price = float(match_price.group(1)) if match_price else None
    t = float(match_time.group(1)) if match_time else None
    return price, t

# ================================
# EXÉCUTION SÉQUENTIEL / OPENMP
# ================================
print("=== Comparaison Séquentiel vs OpenMP ===")
for mode, nthreads in [("sequentiel", 1), ("openmp", 16)]:
    for n in N_values:
        print(f"[{mode}] {n} trajectoires...")
        price, t = run_lsmc(n, nthreads)
        data[mode]["price"].append(price)
        data[mode]["time"].append(t)

# Calcul du speedup (Séquentiel / OpenMP)
speedup = [seq / par if seq and par else None
           for seq, par in zip(data["sequentiel"]["time"], data["openmp"]["time"])]

# ================================
# SCALABILITÉ (1, 2, 4, 8, 16 THREADS)
# ================================
print("\n=== Scalabilité OpenMP ===")
times_scaling = []
for nt in threads_scaling:
    print(f"{nt} thread(s)...")
    _, t = run_lsmc(N_paths_scaling, nt)
    times_scaling.append(t)

t1 = times_scaling[0]
speedup_scaling = [t1 / t if t else None for t in times_scaling]

# ================================
# GRAPHIQUES
# ================================
plt.figure(figsize=(10, 12))

# 1. Convergence du prix
plt.subplot(4, 1, 1)
plt.plot(N_values, data["sequentiel"]["price"], "o-b", label="Séquentiel")
plt.plot(N_values, data["openmp"]["price"], "s-g", label="OpenMP")
plt.title("Convergence du prix estimé (LSMC)")
plt.xlabel("Nombre de trajectoires (N)")
plt.ylabel("Prix estimé")
plt.legend()
plt.grid(True)

# 2. Temps d'exécution
plt.subplot(4, 1, 2)
plt.plot(N_values, data["sequentiel"]["time"], "o--r", label="Séquentiel")
plt.plot(N_values, data["openmp"]["time"], "s--c", label="OpenMP")
plt.title("Temps d'exécution (Séquentiel vs OpenMP)")
plt.xlabel("Nombre de trajectoires (N)")
plt.ylabel("Temps (s)")
plt.legend()
plt.grid(True)

# 3. Speedup par N_paths
plt.subplot(4, 1, 3)
plt.plot(N_values, speedup, "d-m", label="Speedup (Séquentiel / OpenMP)")
plt.title("Gain de performance selon N_paths")
plt.xlabel("Nombre de trajectoires (N)")
plt.ylabel("Facteur d'accélération")
plt.axhline(1.0, color="gray", linestyle="--")
plt.legend()
plt.grid(True)

# 4. Scalabilité réelle (1, 2, 4, 8, 16 threads)
plt.subplot(4, 1, 4)
plt.plot(threads_scaling, speedup_scaling, "o-m", linewidth=2, markersize=8, label="Speedup réel")
for x, y in zip(threads_scaling, speedup_scaling):
    plt.text(x, y + 0.05, f"{y:.2f}", ha="center", fontsize=9)
plt.title(f"Scalabilité OpenMP réelle ({N_paths_scaling} trajectoires)")
plt.xlabel("Nombre de threads")
plt.ylabel("Facteur d'accélération")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
