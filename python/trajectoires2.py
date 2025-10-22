import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Chargement
paths = np.loadtxt("../../lsmc/trajectoires_gbm.csv", delimiter=",")
t = np.linspace(0, 1.0, paths.shape[1])
mean_path = np.mean(paths, axis=0)

plt.figure(figsize=(10,6))

# — Trajectoires individuelles semi-transparentes
for p in paths:
    plt.plot(t, p, color='gray', alpha=0.05, lw=0.7)

# — Densité couleur (heatmap verticale)
all_points = np.vstack([np.repeat(t, paths.shape[0]), paths.T.flatten()])
kde = gaussian_kde(all_points)(all_points)
plt.scatter(all_points[0], all_points[1], c=kde, s=1, cmap='inferno', alpha=0.4)

# — Moyenne
plt.plot(t, mean_path, color='cyan', lw=2, label='Moyenne simulée')

plt.title("Trajectoires simulées (effet densité, export GBM C++)")
plt.xlabel("Temps")
plt.ylabel("Prix de l’actif")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
