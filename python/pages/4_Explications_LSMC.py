import streamlit as st

st.title("Explications théoriques de la méthode")

st.write("""
Cette page présente les idées principales derrière la méthode que nous utilisons pour estimer
le prix d’une option américaine. L’objectif est d’expliquer clairement pourquoi nous avons
besoin de simuler autant de trajectoires et comment la méthode LSMC permet de prendre une
décision d’exercice optimale à chaque instant.
""")

# --------------------------------------------------
# 1) Modèle GBM
# --------------------------------------------------

st.header("1. Modèle d’évolution du prix (GBM)")

st.write("""
Nous supposons que le prix du sous-jacent suit un mouvement brownien géométrique.
Ce modèle décrit une évolution continue, aléatoire et proportionnelle au prix.
Il sert simplement à générer les trajectoires possibles que le marché pourrait suivre.
""")

st.latex(r"dS_t = r S_t\, dt + \sigma S_t\, dW_t")

st.write("""
En pratique, ce modèle nous fournit un grand ensemble de scénarios possibles.
Ces trajectoires servent ensuite de base pour décider à quel moment il est optimal d’exercer l’option.
""")

# --------------------------------------------------
# 2) Exercice optimal d’une option américaine
# --------------------------------------------------

st.header("2. Option américaine et exercice optimal")

st.write("""
Contrairement à une option européenne, qui ne peut être exercée qu’à maturité,
une option américaine peut être exercée à n’importe quel instant.

Cela pose un problème fondamental : comment décider quand exercer ?

À chaque date, pour chaque trajectoire, nous devons comparer deux quantités :
- la valeur obtenue en exerçant immédiatement (le payoff),
- la valeur que l’on pourrait obtenir en continuant et en attendant une date future.

La règle est simple :  
**nous exerçons si la valeur immédiate est meilleure que ce que nous obtiendrions plus tard.**
""")

# --------------------------------------------------
# 3) Régression (approximation de la valeur future)
# --------------------------------------------------

st.header("3. Régression pour estimer la valeur future")

st.write("""
Comme nous ne connaissons pas à l’avance la valeur future, nous devons l’estimer.
La manière proposée par Longstaff et Schwartz consiste à utiliser une régression polynomiale.

L’idée est d’observer, parmi les trajectoires qui ont continué, les valeurs obtenues plus tard,
puis d’ajuster une fonction simple qui relie l’état actuel du sous-jacent à la valeur attendue.
""")

st.latex(r"C(S_t) \approx a_0 + a_1 S_t + a_2 S_t^2")

st.write("""
Cette régression nous donne une approximation de la valeur de continuation.
Elle permet ensuite de comparer proprement “exercer” et “attendre”.
""")

# --------------------------------------------------
# 4) Backward induction
# --------------------------------------------------

st.header("4. Backward induction : remonter le temps")

st.write("""
La méthode LSMC fonctionne en remontant le temps.
Nous commençons à la dernière date de l’option, puis nous revenons vers le début.
À chaque étape :
1. nous regardons quelles trajectoires sont "dans la monnaie",
2. nous estimons leur valeur future grâce à la régression,
3. nous décidons si l’exercice est optimal,
4. nous mettons à jour les cashflows associés.
""")

st.code("""
for t from T-1 to 0:
    sélectionner les trajectoires dans la monnaie
    calculer les cashflows futurs actualisés
    ajuster la régression OLS
    comparer payoff immédiat et valeur de continuation
    appliquer la règle d’exercice optimal
""")

st.write("""
Ce procédé nous permet d’obtenir, pour chaque trajectoire, la meilleure stratégie d’exercice.
En répliquant sur toutes les trajectoires simulées, nous obtenons une estimation du prix de l’option.
""")

st.success("Cette page résume les idées essentielles de la méthode LSMC.")
