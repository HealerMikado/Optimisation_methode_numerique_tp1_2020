#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Fonction de coût et algorithmes pour la régression aux moindres carrés régularisée (L2)."""

import numpy as np
from numpy.random import randn, choice

lambda_ = 0.1

def cost(X, Y, beta):
    """Fonction de coût de la régression aux moindres carrés ordinaire.

    :param X: Coordonnées des n points, chaque ligne correspond à un point
    :type X: numpy.matrix, shape = (n,d)
    :param Y: Valeur de chacun des points
    :type Y: numpy.matrix, shape = (n,1)
    :param beta: Vecteur de paramètres de l'application linéaire
    :type beta: numpy.matrix, shape = (d,1)
    :return: moyenne des erreurs au carré + régularisation L2

    >>> X = np.matrix([[1, 2], [3, 4], [1, 4]])
    >>> Y = np.matrix([[0.1], [0.9], [0.5]])
    >>> beta = np.matrix([[0.2], [0.1]])
    >>> np.round(cost(X, Y, beta), 4)
    0.0417
    """
    return np.mean(np.square((X*beta - Y))) + lambda_ * np.linalg.norm(beta)**2


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
        beta = randn(self.X.shape[1], 1)
        c = cost(self.X, self.Y, beta)
        if c < self.best_cost:
            self.best_beta = beta
            self.best_cost = c
        return self.best_beta

class GradientDescent:
    """Recherche par descente de gradient.

    À chaque pas de temps le point est déplacé dans la direction du gradient de la fonction de coût pondéré par un pas
    de gradient de la forme *a / (1 + b.t)"""

    def __init__(self, X, Y, a=1, b=0):
        """Initialisation de la recherche.

        :param X: Coordonnées des n points, chaque ligne correspond à un point
        :type X: numpy.matrix, shape = (n,d)
        :param Y: Valeur de chacun des points
        :type Y: numpy.matrix, shape = (n,1)
        :param a: paramètre du pas de gadient
        :type a: flottant
        :param b: paramètre du pas de gadient
        :type b: flottant
        """
        self.X = X
        self.Y = Y
        self.a = a
        self.b = b
        self.beta = np.matrix(np.zeros((X.shape[1],1)))
        self.t = 0

    def get_next(self):
        """Prochain point."""
        XX = self.X.T * self.X / self.X.shape[0]      # on divise par le nombre d'exemples pour homogénéiser le comportement
        XY = self.X.T * self.Y / self.X.shape[0]      # on divise par le nombre d'exemples pour homogénéiser le comportement
        self.beta -= self.a / (1 + self.b*self.t) * (2 * (XX*self.beta - XY) + 2 * lambda_ * self.beta)
        self.t += 1
        return self.beta

class StochasticGradientDescent:
    """Recherche par descente de gradient stochastique.

    À chaque pas de temps le point est déplacé dans la direction du gradient de la fonction de coût pondéré par un pas
    de gradient de la forme *a / (1 + b.t).

    On ne calcule pas le gradient exact, mais celui associé à un "mini-batch" d'exemples.

    """

    def __init__(self, X, Y, mb_size=1, a=1, b=0):
        """Initialisation de la recherche.

        :param X: Coordonnées des n points, chaque ligne correspond à un point
        :type X: numpy.matrix, shape = (n,d)
        :param Y: Valeur de chacun des points
        :type Y: numpy.matrix, shape = (n,1)
        :param a: paramètre du pas de gadient
        :type a: flottant
        :param b: paramètre du pas de gadient
        :type b: flottant
        """
        self.X = X
        self.Y = Y
        self.mb_size = mb_size
        self.a = a
        self.b = b
        self.beta = np.matrix(np.zeros((X.shape[1],1)))
        self.t = 0

    def get_next(self):
        """Prochain point."""
        inds = choice(self.X.shape[0], self.mb_size, replace=False)
        XX = self.X[inds,:].T * self.X[inds,:] / self.mb_size
        XY = self.X[inds,:].T * self.Y[inds,:] / self.mb_size
        self.beta -= self.a / (1 + self.b*self.t) * (2 * (XX*self.beta - XY) + 2 * lambda_ * self.beta)
        self.t += 1
        return self.beta

if __name__ == "__main__":
    import doctest
    doctest.testmod()
