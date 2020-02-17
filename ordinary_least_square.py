#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Fonction de coût et algorithmes pour la régression aux moindres carrés ordinaire."""

import numpy as np
from numpy.random import randn, choice

def cost(X, Y, beta):
    """Fonction de coût de la régression aux moindres carrés ordinaire.

    :param X: Coordonnées des n points, chaque ligne correspond à un point
    :type X: numpy.matrix, shape = (n,d)
    :param Y: Valeur de chacun des points
    :type Y: numpy.matrix, shape = (n,1)
    :param beta: Vecteur de paramètres de l'application linéaire
    :type beta: numpy.matrix, shape = (d,1)
    :return: MSE : moyenne des erreurs au carré

    >>> X = np.matrix([[1, 2], [3, 4], [1, 4]])
    >>> Y = np.matrix([[0.1], [0.9], [0.5]])
    >>> beta = np.matrix([[0.2], [0.1]])
    >>> np.round(cost(X, Y, theta), 4)
    0.0367
    """
    return np.mean(np.square((X*beta - Y)))


class RandomSearch:
    """Recherche aléatoire du point optimal.

    À chaque pas de temps, un point aléatoire est essayé.
    - s'il est meilleur que le point au pas de temps précédent, on le concerve et on le retourne.
    - sinon, on retourne le meilleur point rencontré jusquà présent."""

    def __init__(self, X, Y):
        """Initialisation de la recherche.

        :param X: Coordonnées des n points, chaque ligne correspond à un point
        :type X: numpy.matrix, shape = (n,d)
        :param Y: Valeur de chacun des points
        :type Y: numpy.matrix, shape = (n,1)
        """
        self.X = X
        self.Y = Y
        self.best_cost = np.infty

    def get_next(self):
        """Prochain point."""
        theta = randn(self.X.shape[1], 1)
        c = cost(self.X, self.Y, theta)
        if c < self.best_cost:
            self.best_beta = theta
            self.best_cost = c
        return self.best_beta



if __name__ == "__main__":
    import doctest
    doctest.testmod()
