#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Compare les algorithmes"""

import data
import ordinary_least_square as ols
from manager import Manager

X, Y = data.get_data()


#=======================================================
# Comparons différentes exécutions de la recherche aléatoire
#=======================================================
if False:
    manager = Manager(X, Y, ols.cost)

    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire", 1000, 10 ** (-5))
    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire (bis)", 1000, 10 ** (-5))
    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire (ter)", 1000, 10 ** (-5))
    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire (quat)", 1000,10 ** (-5))


    manager.show_plots()


#=======================================================
# Comparons différents pas de la méthode de gradient
#=======================================================
if True:
    manager = Manager(X, Y, ols.cost)

    #manager.run(ols.GradientDescent(X, Y, 10**-1),
    #            "descente de gradient, 10^-1", 5000, 10 ** (-5))
    manager.run(ols.GradientDescent(X, Y, 10**-2),
                "descente de gradient, 10^-2", 5000, 10 ** (-4))
    manager.run(ols.GradientDescent(X, Y, 10**-3),
                "descente de gradient, 10^-3", 5000, 10 ** (-4))
    manager.run(ols.GradientDescent(X, Y, 10**-4),
                "descente de gradient, 10^-4", 5000, 10 ** (-4))
    manager.run(ols.GradientDescent(X, Y, 10**-4),
                "descente de gradient, 10^-4", 5000, 10 ** (-5))

    manager.show_plots()


#=======================================================
# Comparons différents pas décroissants de la méthode de gradient
#=======================================================
if False:
    manager = Manager(X, Y, ols.cost)

    manager.run(ols.GradientDescent(X, Y, 10**-2), "descente de gradient, a=10^-2, b=0", 2000, 10 ** (-5))
    manager.run(ols.GradientDescent(X, Y, 10**-2, 10**-4), "descente de gradient, a=10^-2, b=10^-4", 2000, 10 ** (-5))
    manager.run(ols.GradientDescent(X, Y, 10**-2, 0.01), "descente de gradient, a=10^-2, b=10^-2", 2000, 10 ** (-5))
    manager.run(ols.GradientDescent(X, Y, 10**-2, 1), "descente de gradient, a=10^-2, b=1", 2000, 10 ** (-5))

    manager.show_plots()

#=======================================================
# Comparons différents pas décroissants de la méthode de gradient
#=======================================================
if False:
    manager = Manager(X, Y, ols.cost)

    manager.run(ols.GradientDescent(X, Y, 10**-2), "descente de gradient, a=10^-2, b=0", 2000, 10 ** (-5))
    manager.run(ols.GradientDescent(X, Y, 10**-1, 0.01), "descente de gradient, a=10^-1, b=10^-2", 2000, 10 ** (-5))
    manager.run(ols.GradientDescent(X, Y, 10**-1, 1), "descente de gradient, a=10^-1, b=1", 2000, 10 ** (-5))

    manager.show_plots()

#=======================================================
# Vérifions que la méthode de gradient stochastique est stochastique
#=======================================================
if False:
    manager = Manager(X, Y, ols.cost)

    manager.run(ols.StochasticGradientDescent(X, Y, 1, 10**-2), "SGD, mini-batch=1, a=10^-2", 2000, 10 ** (-5))
    manager.run(ols.StochasticGradientDescent(X, Y, 1, 10**-2), "SGD, mini-batch=1, a=10^-2 (bis)", 2000, 10 ** (-5))
    manager.run(ols.StochasticGradientDescent(X, Y, 1, 10**-2), "SGD, mini-batch=1, a=10^-2 (ter)", 2000, 10 ** (-5))

    manager.show_plots()


#=======================================================
# Comparons l'effet de la taille du mini-batch pour la méthode de gradient stochastique
#=======================================================
if False:
    manager = Manager(X, Y, ols.cost)

    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire", 2000, 10 ** (-5))
    manager.run(ols.GradientDescent(X, Y, 10**-2), "descente de gradient, a=10^-2", 2000, 10 ** (-5))
    manager.run(ols.StochasticGradientDescent(X, Y, 100, 10**-2), "SGD, mini-batch=100, a=10^-2", 2000, 10 ** (-5))
    manager.run(ols.StochasticGradientDescent(X, Y, 10, 10**-2), "SGD, mini-batch=10, a=10^-2", 2000, 10 ** (-5))
    manager.run(ols.StochasticGradientDescent(X, Y, 1, 10**-2), "SGD, mini-batch=1, a=10^-2", 2000, 10 ** (-5))

    manager.show_plots()


#=======================================================
# Et avec la fonction régularisée en norme 2 ?
#=======================================================
if False:
    manager = Manager(X, Y, rls.cost)
    manager.run(rls.RandomSearch(X, Y), "aléa", 2000, 10 ** (-5))
    manager.run(rls.GradientDescent(X, Y, 10**-2), "GD 10^-2", 2000, 10 ** (-5))
    manager.run(rls.GradientDescent(X, Y, 10**-3), "GD 10^-3", 2000, 10 ** (-5))
    manager.run(rls.GradientDescent(X, Y, 10 ** -2, 0.01), "GD 10^-2 10^-2", 2000, 10 ** (-5))
    manager.run(rls.GradientDescent(X, Y, 10 ** -2, 0.001), "GD 10^-2 10^-3", 2000, 10 ** (-5))
    manager.show_plots()
if False:
    manager = Manager(X, Y, rls.cost)
    manager.run(rls.GradientDescent(X, Y, 10**-2), "GD 10^-2", 2000, 10 ** (-5))
    manager.run(rls.StochasticGradientDescent(X, Y, 100, 10**-2), "SGD 100 10^-2", 10000, 10 ** (-5))
    manager.run(rls.StochasticGradientDescent(X, Y, 10, 10**-2), "SGD 10 10^-2", 10000, 10 ** (-5))
    manager.run(rls.StochasticGradientDescent(X, Y, 1, 10**-2), "SGD 1 10^-2", 20000, 10 ** (-5))
    manager.show_plots()
if False:
    manager = Manager(X, Y, rls.cost)
    manager.run(rls.StochasticGradientDescent(X, Y, 100, 10**-2), "SGD 100 10^-2", 10000, 10 ** (-5))
    manager.run(rls.StochasticGradientDescent(X, Y, 100, 10**-2, 0.001), "SGD 100 10^-2 10^-3", 100000, 10 ** (-5))
    manager.run(rls.StochasticGradientDescent(X, Y, 100, 10**-2, 0.01), "SGD 100 10^-2 10^-2", 100000, 10 ** (-5))
    manager.run(rls.StochasticGradientDescent(X, Y, 100, 10**-2, 0.1), "SGD 100 10^-2 10^-1", 50000, 10 ** (-5))
    manager.show_plots()

#=======================================================
# Et avec la fonction régularisée en norme 1 ?
#=======================================================
if False:
    manager = Manager(X, Y, lasso.cost)


if False:
    manager = Manager(X, Y, lasso.cost)


if False:
    manager = Manager(X, Y, lasso.cost)

