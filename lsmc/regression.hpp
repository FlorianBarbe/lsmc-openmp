/**
 * regression.hpp
 * Régression polynomiale OLS de degré 2
 */
#pragma once
#include <vector>

namespace Regression {
    // Calcule les coefficients d’une régression polynomiale (degré 2)
    std::vector<double> ols_poly2(const std::vector<double>& x, const std::vector<double>& y);
    // Prédit les valeurs selon les coefficients
    std::vector<double> predict_poly2(const std::vector<double>& x, const std::vector<double>& coef);
}
//Regression OLS (Ordinary Least Squares) est une méthode statistique utilisée pour estimer les coefficients d'un modèle linéaire.
//forme générale du modèle:
//yi=β0+β1*xi1x+β2*xi2x+...+βn*xin+εi
//avec : 
//yi la variable dépendante
//xij les variables explicatives
//βj les coefficients à estimer 
//εi le terme d'erreur aléatoire
//Principe : OLS choisit les coefficients beta chapeau qui minimisent la somme des carrés des résidues, ie:
//min{beta}(MSE(Y))

//Hypothèses principales:
//Linéarité: La relation entre les variables dépendantes et indépendantes est linéaire.
//Espérance du résidu nulle: E[εi]=0.
//Homoscedasticité: Variance constante des erreurs
//Indpdce des observations.
//Pas de multicolinéarité parfaite entre variables explicatives 
// (ie: les variables explicatives ne doivent pas être parfaitement coréllées entre elles.
//Sinon , XTX n'est pas inversible, donc beta chapeau = (XTX)^-1XT Y n'est pas calculable. ) (ex : impossible de distinguer l'effet propre de x1 et x si x2=2*x1)

//Résultat : Si ces hypothèses sont respectées, les estimateurs OLS sont non biaisés, efficients et consistants.
//Efficient : Parmi tous les estimateurs non biaisés, l'OLS a la plus petite variance -> il est le plus précis possible sans introduire de biais. (voir thm de Gauss-Markov : OLS est le meilleur estimateur linéaire non biaisé (BLUE, Best Linear Unbiased Estimator).
//Consistant : A mesure que la taille de l'échantillon augmente, les estimateurs OLS convergent vers les vraies valeurs des paramètres.
//Beta chapeau -> Beta pour n-> infini.Donc la variance tend vers 0 et l'estimateur est de plus en plus précis.