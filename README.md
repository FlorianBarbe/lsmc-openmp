============================================================
 PROJET LSMC - PRICING D'OPTIONS AMÉRICAINES EN C++ (OpenMP)
============================================================

1. OBJECTIF
------------
Ce projet implémente la méthode de Longstaff-Schwartz (LSMC)
pour estimer le prix d'une option américaine par simulation
Monte Carlo et régression linéaire. La simulation utilise un
mouvement brownien géométrique (GBM). L'exécution est
parallélisée avec OpenMP, et les résultats sont exportés
vers un fichier CSV pour visualisation Python.

Modules :
 - Simulation Monte Carlo (GBM)
 - Régression OLS
 - Backward Induction
 - Parallélisation OpenMP
 - Export CSV et visualisation Python

------------------------------------------------------------

2. STRUCTURE DU PROJET
----------------------
lsmc/
│
├── include/
│   ├── gbm.hpp          : déclaration de la classe GBM
│   ├── regression.hpp   : fonctions de régression OLS
│   ├── rng.hpp          : générateur de nombres aléatoires
│   └── lsmc.hpp         : logique du modèle Longstaff-Schwartz
│
├── src/
│   ├── gbm.cpp          : implémentation de la classe GBM
│   ├── regression.cpp   : régression par moindres carrés
│   ├── rng.cpp          : tirages normaux standard
│   ├── lsmc.cpp         : algorithme LSMC complet
│   └── main.cpp         : point d’entrée du programme
│
├── x64/Debug/
│   └── visualisation.py : script de visualisation Python
│
└── output/
    └── trajectoires_gbm.csv : trajectoires simulées (export C++)

------------------------------------------------------------

3. DESCRIPTION DES MODULES
---------------------------

a) RNG (rng.hpp / rng.cpp)
   - Gère les tirages aléatoires selon une loi normale.
   - Utilisé dans GBM pour générer les incréments Brownien.
   - Fournit : double normal();

b) GBM (gbm.hpp / gbm.cpp)
   - Simule le mouvement brownien géométrique :
       S_{t+dt} = S_t * exp((r - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z)
   - Fonctions :
       simulate()            : une trajectoire unique
       simulatePaths()       : N trajectoires (parallèle OpenMP)
       exportCSV()           : export vers un fichier CSV
   - Le CSV est ensuite visualisé dans Python.

c) Regression (regression.hpp / regression.cpp)
   - Implémente une régression OLS (Ordinary Least Squares)
     pour estimer la valeur conditionnelle à chaque étape
     du backward induction.
   - Utilisé dans LSMC.

d) LSMC (lsmc.hpp / lsmc.cpp)
   - Implémente la méthode Longstaff-Schwartz :
       1. Génère N trajectoires de prix (GBM)
       2. Calcule les payoffs à chaque date
       3. Réalise des régressions backward pour estimer
          la continuation value
       4. Décide d’exercer ou non l’option
   - Retourne le prix moyen de l’option américaine.

e) main.cpp
   - Configure les paramètres du modèle (S0, K, r, sigma, T)
   - Affiche l’état du parallélisme (OpenMP activé ou non)
   - Simule les trajectoires (GBM::simulatePaths)
   - Exporte les résultats vers CSV
   - Calcule et affiche le prix du put américain
   - Mesure le temps d’exécution
   - Compare les modes séquentiel / parallèle

------------------------------------------------------------

4. PARALLÉLISATION (OpenMP)
----------------------------
Les boucles de simulation Monte Carlo sont parallélisées :

    #pragma omp parallel for

Chaque thread génère ses propres tirages aléatoires
(RNG local). Le gain observé dépend du matériel :
 - Speedup réel : environ x1.6 à 4 threads
 - Saturation au-delà de 8 threads (limite mémoire)
 - Bénéfice limité par la bande passante RAM

------------------------------------------------------------

5. EXPORT ET VISUALISATION PYTHON
---------------------------------
Le programme C++ écrit un fichier :
    lsmc/output/trajectoires_gbm.csv

Ce fichier contient N_paths lignes (une par trajectoire)
et N_steps colonnes (prix à chaque instant).

Script Python associé : visualisation.py

   - Charge le CSV :
        paths = np.loadtxt("../../lsmc/trajectoires_gbm.csv", delimiter=",")
   - Trace les trajectoires (grises, transparentes)
   - Affiche la moyenne simulée (bleue)
   - Ajoute une densité de probabilité (heatmap)
   - Produit des graphiques lisibles et scientifiques

------------------------------------------------------------

6. ANALYSE DE PERFORMANCE
-------------------------
Mesures principales :
 - Speedup moyen ≈ 1.6× à 4 threads
 - Gain marginal positif jusqu’à 4 threads
 - Saturation de performance ensuite
 - Temps d’exécution divisé par 1.5 à 1.6 sur CPU 8 cœurs

Limitations :
 - Calcul limité par les accès mémoire
 - OpenMP utile pour N_paths élevés (> 5000)

------------------------------------------------------------

7. FICHIERS PYTHON D’ANALYSE
----------------------------
visualisation.py       : Affiche les trajectoires simulées.
analyse_convergence.py : Compare séquentiel / OpenMP :
                         - Prix estimé
                         - Temps d’exécution
                         - Speedup
                         - Scalabilité 3D et heatmap.

------------------------------------------------------------

8. UTILISATION
--------------
Compilation :
 - Ouvrir le projet Visual Studio
 - Mode Debug ou Release x64
 - Compiler le projet "lsmc"

Exécution :
 - Lancer depuis Visual Studio (F5)
 - Le fichier CSV est créé automatiquement
 - Lancer ensuite visualisation.py pour tracer les courbes

------------------------------------------------------------

9. AUTEUR
---------
Projet développé par :
 - Florian Barbe et Narjisse
Dans le cadre du module LSMC (Méthodes de pricing)
École Centrale de Nantes - 2025
