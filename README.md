**PROJET LSMC - PRICING D'OPTIONS AMÉRICAINES EN C++ (OpenMP)**

**1. OBJECTIF**  
Ce projet implémente la méthode de Longstaff–Schwartz (LSMC) pour estimer le prix d’une option américaine par simulation Monte Carlo avec régression OLS.  
Les trajectoires sont simulées sous un mouvement brownien géométrique (GBM).  
La simulation est parallélisée avec OpenMP et les résultats sont exportés vers des CSV exploitables en Python/Streamlit.

Modules :
- Simulation Monte Carlo (GBM)
- Régression OLS
- Backward Induction
- Parallélisation OpenMP
- Export CSV et visualisation Python
- Backend GPU expérimental (CUDA)

------------------------------------------------------------

**2. STRUCTURE DU PROJET**

lsmc-openmp/  
&nbsp;&nbsp;&nbsp;&nbsp;include/        # Headers : GBM, LSMC, régression OLS, RNG  
&nbsp;&nbsp;&nbsp;&nbsp;src/            # Implémentations CPU + OpenMP + main.cpp  
&nbsp;&nbsp;&nbsp;&nbsp;cuda/           # Version GPU CUDA (kernels et expérimentations)  
&nbsp;&nbsp;&nbsp;&nbsp;python/         # Interface Streamlit et scripts d'analyse  
&nbsp;&nbsp;&nbsp;&nbsp;docs/           # Documentation, rapports, images  
&nbsp;&nbsp;&nbsp;&nbsp;.gitignore      # Ignore CSV, binaires, x64/, Debug/, etc.  
&nbsp;&nbsp;&nbsp;&nbsp;lsmc.sln        # Solution Visual Studio  
&nbsp;&nbsp;&nbsp;&nbsp;README.md       # Documentation principale  

------------------------------------------------------------

**3. DESCRIPTION DES MODULES**

RNG (rng.hpp / rng.cpp)  
- Génère des tirages normaux.  
- Utilisé dans GBM pour simuler les incréments brownien.

GBM (gbm.hpp / gbm.cpp)  
- Simulation du mouvement brownien géométrique.  
- Fonctions : simulate(), simulatePaths(), exportCSV().

Regression (regression.hpp / regression.cpp)  
- Régression polynomiale OLS.  
- Approxime la continuation value dans le backward LSMC.

LSMC (lsmc.hpp / lsmc.cpp)  
- Simulation GBM + backward induction + décisions d'exercice.

main.cpp  
- Configure les paramètres, exécute séquentiel/OpenMP, écrit les CSV.

------------------------------------------------------------

**4. HISTORIQUE DE DÉVELOPPEMENT**

**22 octobre 2024**  
- Création du projet  
- Création du dépôt et structure initiale.  
- Première version de GBM, LSMC et scripts Python.

**23 octobre 2024**  
- Ajustements  
- Correction du chemin des CSV, nettoyage initial.

**6 novembre 2024**  
- Documentation OLS  
- Ajout de la théorie LSMC dans le README.  
- Documentation GBM/LSMC/Regression.

**9 novembre 2024**  
- Suppression complète  
- Suppression de l'ancien code et reconstruction prévue.

**10 novembre 2024**  
- Reconstruction  
- Recréation propre de src/, include/, nouveau pipeline CSV.

**11 novembre 2024**  
- Reset Git  
- Force-push pour remettre le dépôt au propre.

**18–19 novembre 2024**  
- Refonte Streamlit  
- Réécriture complète de l’interface UI.  
- Mise à jour des CSV.

**20 novembre 2024**  
- Export CSV robuste  
- Export sécurisé via fichier temporaire + rename atomique.

**24 novembre 2024**  
- Stabilisation  
- Corrections diverses et nettoyage.

**24 novembre 2025**  
- Mise à jour Python et documentation  
- Organisation des pages Streamlit.  
- Commentaires GBM enrichis.

**29 novembre 2025**
- Implémentation CUDA  
  - Ajout du backend GPU complet dans `cuda/lsmc_cuda.cu`.  
  - Écriture des kernels :
      • simulatePathsKernel : simulation GBM entièrement sur GPU  
      • buildNormalEquationsKernel : construction de AᵀA et Aᵀy via atomicAdd  
      • updateCashflowsKernel : décision exercice/continuation et mise à jour des cashflows  
      • solve3x3 (device) : résolution d’un système linéaire 3×3 pour l’OLS  
  - Mise en place des allocations GPU, transferts host/device, synchronisations et nettoyage mémoire.  
  - Intégration complète dans Visual Studio :
      • ajout du dossier cuda/ dans .vcxproj  
      • ajout des filtres pour l’organisation dans l’explorateur de solutions  

- Nettoyage du dépôt  
  - Le push échouait systématiquement à ~72 %, avec :  
    • “RPC failed”  
    • “HTTP 408”  
    • “remote end hung up unexpectedly”  
  - Analyse : GitHub refusait un fichier >100 Mo (`trajectoires_gbm.csv` ~213 Mo) encore présent dans l’historique Git (même si supprimé physiquement).  
  - Solutions appliquées :
      • mise en place d’un `.gitignore` complet pour bloquer les CSV lourds, les x64/, binaires, Debug/, etc.  
      • purge de tout l’historique Git avec `git filter-repo` pour supprimer définitivement les blobs >100 Mo  
      • suppression complète des objets résiduels dans `.git/objects`  
      • réinstallation propre de Git et de Git Credential Manager Core (les credentials corrompus bloquaient certains push)  
      • force-push final pour publier l’historique propre sur GitHub  
  - Résultat : dépôt allégé, propre, compatible GitHub, plus aucun blocage.
