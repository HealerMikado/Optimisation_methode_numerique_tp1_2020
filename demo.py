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
if True:
    manager = Manager(X, Y, ols.cost)

    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire", 500, 10 ** (-5))
    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire (bis)", 1000, 10 ** (-5))
    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire (ter)", 1000, 10 ** (-5))
    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire (quat)", 1000,10 ** (-5))


    manager.show_plots()


#=======================================================
# Comparons différents pas de la méthode de gradient
#=======================================================
if True:
    manager = Manager(X, Y, ols.cost)



#=======================================================
# Comparons différents pas décroissants de la méthode de gradient
#=======================================================
if False:
    manager = Manager(X, Y, ols.cost)


#=======================================================
# Comparons différents pas décroissants de la méthode de gradient
#=======================================================
if False:
    manager = Manager(X, Y, ols.cost)


#=======================================================
# Vérifions que la méthode de gradient stochastique est stochastique
#=======================================================
if False:
    manager = Manager(X, Y, ols.cost)



#=======================================================
# Comparons l'effet de la taille du mini-batch pour la méthode de gradient stochastique
#=======================================================
if False:
    manager = Manager(X, Y, ols.cost)



#=======================================================
# Et avec la fonction régularisée en norme 2 ?
#=======================================================
if False:
    manager = Manager(X, Y, rls.cost)


if False:
    manager = Manager(X, Y, rls.cost)


if False:
    manager = Manager(X, Y, rls.cost)


#=======================================================
# Et avec la fonction régularisée en norme 1 ?
#=======================================================
if False:
    manager = Manager(X, Y, lasso.cost)


if False:
    manager = Manager(X, Y, lasso.cost)


if False:
    manager = Manager(X, Y, lasso.cost)

