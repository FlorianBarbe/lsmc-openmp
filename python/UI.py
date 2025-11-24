import streamlit as st

st.set_page_config(page_title="LSMC OpenMP", layout="wide")

st.title("Projet LSMC — Pricing d’options américaines en C++ (OpenMP)")

st.write("""
Cette page présente notre application développée dans le cadre du projet PRV/INFO-RV, 
dans la partie consacrée aux méthodes numériques avant l’étude des approches par Deep Learning.  
Nous y implémentons la méthode de Longstaff–Schwartz (LSMC) en C++ avec parallélisation OpenMP, 
et nous utilisons Streamlit pour structurer l’interface et visualiser les résultats.

---

## Objectif général du projet

Nous cherchons à calculer le prix d’une option américaine, c’est-à-dire un actif dérivé exerçable à tout moment avant l’échéance.
Pour situer le cadre :

- une option **européenne** ne peut être exercée qu’à maturité ;
- une option **américaine** peut l’être à n’importe quelle date, ce qui complique fortement le calcul du prix.

Le but de ce projet est de comprendre le principe de la méthode LSMC, de la coder proprement, 
d’explorer son comportement numérique, puis d’analyser la contribution de la parallélisation OpenMP.

---

## Contexte : modèles de marché et prix des options

Nous travaillons dans le modèle classique du mouvement brownien géométrique (GBM), 
le modèle utilisé dans le cadre de Black, Scholes et Merton (1973).  
Leur travail a introduit une modélisation rigoureuse du prix des actifs, des équations différentielles stochastiques, 
et a donné une formule analytique pour le prix des options européennes.  

Cependant, **il n’existe pas de formule fermée générale pour les options américaines** dans ce cadre.
Pour ces dernières, il faut utiliser des méthodes numériques : modèles binomiaux, différences finies, ou méthodes Monte-Carlo.

Le Monte-Carlo standard ne permet pas de traiter l’exercice anticipé.  
C’est précisément ce que permet la méthode de **Longstaff–Schwartz (2001)**, grâce à une régression polynomiale 
qui estime la valeur de continuation le long des trajectoires futures.

---

## Principe de la méthode LSMC

Nous appliquons la version standard de l’algorithme, qui repose sur :

1. la simulation de nombreuses trajectoires du GBM ;
2. la descente backward dans le temps ;
3. une régression polynomiale (OLS) pour approximer la valeur de continuation ;
4. la décision d’exercice ou de continuation à chaque date ;
5. la projection finale à l’instant initial pour obtenir le prix.

La méthode est naturellement parallélisable, notamment au niveau de la simulation des trajectoires et du calcul local des régressions.
Notre code C++ exploite OpenMP pour mesurer le gain de performance lié au parallélisme.

---

## Ce que notre application permet de faire

Nous avons organisé l’interface en plusieurs pages permettant de :

### 1. Lancer une simulation complète
- génération de trajectoires GBM en C++ ;
- calcul backward LSMC avec régression ;
- version séquentielle et version OpenMP ;
- export automatique des fichiers CSV.

### 2. Visualiser les trajectoires
- affichage d’un ensemble de trajectoires simulées ;
- comparaison avec l’espérance analytique du modèle GBM ;
- distribution, variance et volatilité.

### 3. Étudier les performances
- comparaison séquentiel / OpenMP ;
- analyse du speedup ;
- influence du nombre de trajectoires et du nombre de pas de temps.

### 4. Reprendre les éléments théoriques
- rappel du modèle GBM ;
- fondements du modèle de Black–Scholes–Merton ;
- principe de l’exercice optimal ;
- structure détaillée de la méthode de Longstaff–Schwartz.

---

## Navigation

Le menu latéral permet d’accéder à chacune des pages pour :
- comprendre la méthode,
- lancer une simulation complète,
- visualiser les trajectoires,
- analyser les performances de notre implémentation.

Nous présentons ici la démarche complète que nous avons suivie pour étudier une méthode numérique fondamentale avant d’aborder les approches plus avancées basées sur le Deep Learning.
""")

st.info("Utilisez la barre latérale pour accéder aux différentes sections de l'application.")
