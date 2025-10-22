# 🎯 Design technique – Projet LSMC (Longstaff-Schwartz Monte Carlo)

Ce document décrit la conception technique du projet **LSMC – OpenMP**, développé en C++ et Python.  
L’objectif du projet est de calculer le **prix d’une option américaine** à l’aide de la méthode **Longstaff–Schwartz Monte Carlo (LSMC)**, parallélisée avec **OpenMP** pour accélérer la simulation des trajectoires.  
Le projet combine des composants de simulation numérique, de régression statistique et de visualisation scientifique.

---

## 1. Objectif du module

Le projet vise à :
- simuler un grand nombre de trajectoires d’un actif financier suivant un **Mouvement Brownien Géométrique (GBM)**,  
- estimer le **prix d’un put américain** par la méthode **Longstaff–Schwartz**, en utilisant des régressions polynomiales pour approximer la valeur de continuation,  
- exploiter **OpenMP** pour paralléliser les calculs sur plusieurs cœurs,  
- comparer les résultats à un modèle de référence (formule de Black–Scholes ou arbre binomial),  
- et fournir des outils de **visualisation Python** pour analyser la convergence et les performances du modèle.

Ce module constitue la base d’une chaîne complète de pricing, depuis la génération aléatoire jusqu’à la production de figures exploitables dans le rapport final.

---

## 2. Architecture logicielle

```mermaid
graph TD
    A[main.cpp — point d'entrée du programme]
    B[GBM — simulation du mouvement brownien géométrique]
    C[LSMC — algorithme de Longstaff-Schwartz pour option américaine]
    D[Regression — régression polynomiale OLS]
    E[RNG — générateur aléatoire Box-Muller / Mersenne Twister]
    F[Export CSV — écriture des trajectoires simulées]
    G[Python scripts — analyse et visualisation]

    A -->|Configure paramètres S0, K, r, σ, T, N_steps, N_paths| B
    A -->|Appelle la fonction de pricing| C
    C -->|Utilise trajectoires de prix| B
    C -->|Réalise régressions backward| D
    D -->|Renvoie coefficients de régression| C
    B -->|Utilise RNG pour tirages normaux| E
    B -->|Produit N_paths trajectoires| F
    F -->|trajectoires_gbm.csv| G

