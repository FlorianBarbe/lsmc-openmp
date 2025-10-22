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
    A[main.cpp<br/><small>Point d'entrée du programme</small>]
    B[GBM<br/><small>Simulation du Mouvement Brownien Géométrique</small>]
    C[LSMC<br/><small>Algorithme de Longstaff-Schwartz<br/>(pricing option américaine)</small>]
    D[Regression<br/><small>Régression polynomiale (OLS)</small>]
    E[RNG<br/><small>Générateur aléatoire (Box-Muller, Mersenne Twister)</small>]
    F[Export CSV<br/><small>Sortie des trajectoires simulées</small>]
    G[Python scripts<br/><small>Analyse et visualisation</small>]

    A -->|Configure paramètres S₀, K, r, σ, T, N_steps, N_paths| B
    A -->|Appelle la fonction de pricing| C
    C -->|Demande les trajectoires simulées| B
    C -->|Effectue les régressions backward| D
    D -->|Renvoie les coefficients β| C
    B -->|Utilise le générateur normal N(0,1)| E
    B -->|Produit N_paths trajectoires| F
    F -->|Sauvegarde sous trajectoires_gbm.csv| G
    G -->|Affiche les trajectoires et la moyenne| G
