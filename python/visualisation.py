import numpy as np
import matplotlib.pyplot as plt

paths = np.loadtxt("../../lsmc/trajectoires_gbm.csv", delimiter=",")
t = np.linspace(0, 1.0, paths.shape[1])

plt.figure(figsize=(10,6))
for p in paths:
    plt.plot(t, p, color='red', alpha=0.05, lw=0.7)
plt.plot(t, np.mean(paths, axis=0), color='blue', lw=2, label='Moyenne simulée')
plt.title("Trajectoires simulées (export GBM C++)")
plt.xlabel("Temps")
plt.ylabel("Prix de l’actif")
plt.legend()
plt.grid(True)
plt.show()
