import streamlit as st

st.title("📚 Explications théoriques — Méthode LSMC")

st.header("1. Modèle GBM")
st.latex(r"dS_t = rS_t\,dt + \sigma S_t\, dW_t")

st.write("""
On simule des trajectoires selon un mouvement brownien géométrique.
""")

st.header("2. Option américaine et exercice optimal")
st.write("""
À chaque date, on compare :
- payoff immédiat
- valeur de continuation (estimée par régression)

Décision : **exercer si payoff ≥ continuation value**.
""")

st.header("3. Régression OLS")
st.latex(r"C(S_t) \approx a_0 + a_1 S_t + a_2 S_t^2")

st.write("""
Régression polynomiale d'ordre 2 (Longstaff & Schwartz).
""")

st.header("4. Backward Induction")
st.code("""
for t from T-1 to 0:
    select paths in-the-money
    compute discounted cashflows
    regression OLS
    compare payoff vs continuation value
    update cashflows
""")

st.success("Explications détaillées du modèle LSMC.")
