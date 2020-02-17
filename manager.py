#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Classe pour gérer les runs et les logs."""

import numpy as np
from time import time
from math import floor
import matplotlib.pyplot as plt


def logarithmic_indices(stop, n):
    """Retourne *n* indices répartis de façon logarithmique entre 0 et *stop-1*.
    """
    return np.unique(
        [floor(np.exp(i / (n - 1) * np.log(stop))) - 1 for i in range(n)])


class Manager:
    """Execute les algorithmes proposés et affiche leurs résultats."""

    def __init__(self, X, Y, cost_function):
        """Initialisation du gestionnaire.

        :param X: Coordonnées des n points, chaque ligne correspond à un point
        :type X: numpy.matrix, shape = (n,d)
        :param Y: Valeur de chacun des points
        :type Y: numpy.matrix, shape = (n,1)
        :param cost_function: Fonction de coût (à minimiser)
        :type Y: fonction de R^d dans R
        """
        self.X = X
        self.Y = Y
        self.cost_function = cost_function
        self.logs = {}

    def run(self, algo, algo_label, iter_max, threshold):
        """Fait appel à algo.get_next() pour optimiser la fonction de coût *cost*.

        Le run fait au plus *iter_max* appels à algo.get_next(), et s'arrête si la différence de coût entre une
        itération et l'itération suivante est inférieure à *threshold*.

        :param algo: algorithme à faire tourner pour minimiser la fonction *cost*.
        :type algo: objet implémenant get_next()
        :param algo_label: Nom associé à l'algorithme.
        :type algo_label: chaîne de caractères
        :param iter_max: nombre maximum d'itérations à effectuer.
        :type iter_max: int
        :param threshold: seuil entre deux coûts successifs en dessous duquel les itérations s'arrêtent
        """
        costs = []
        times = []
        i = 0
        cost = threshold + 1
        while i < iter_max and cost > threshold:
            i += 1
            t0 = time()
            theta = algo.get_next()
            t1 = time()
            cost = self.cost_function(self.X, self.Y, theta)
            costs.append(cost)
            times.append(t1 - t0)
            if i % (iter_max // 10) == 0:
                print("iteration %d: %.5f" % (i, costs[-1]))

            if self.a_converger(costs, threshold):
                print("A convergé !!!")
        self.logs[algo_label] = {'costs': np.array(costs),
                                 'times': np.cumsum(times)}

    def a_converger(self,costs, threshold):
        """
        Détermine si l'algorithme a convergé. On a convergé si le coût
        moyen des 10 dernières itérations est < seuil
        :return: True si moyenne des 10 dernière iter < seuil. False sinon
        """
        return np.mean(costs[-10:]) < threshold



    def show_plots(self):
        """Affiche les résultats des runs.
        """
        plt.subplot(2, 2, 1)
        self.plot_costs(given_time=False, logscale=False, show=False)
        plt.subplot(2, 2, 2)
        self.plot_costs(given_time=True, logscale=False, show=False)
        plt.subplot(2, 2, 3)
        self.plot_costs(given_time=False, logscale=True, show=False)
        plt.subplot(2, 2, 4)
        self.plot_costs(given_time=True, logscale=True, show=False)
        plt.show()

    def plot_costs(self, given_time=False, logscale=False, show=True):
        """Affiche l'évolution de la fonction de coût en fonction de l'itération / du temps de calcul.

        :param given_time: si *given_time* vaut vrai, le coût est affiché en fonction du temps de calcul.
        :type given_time: booléen
        :param logscale: si *logscale* vaut vrai, les courbes sont affichées en échelle logarithmique.
        :type logscale: booléen
        """
        if show: plt.clf()
        for key, val in self.logs.items():
            inds = logarithmic_indices(val['costs'].shape[0], 100)
            if given_time:
                xs = val['times'][inds]
            else:
                xs = inds + 1
            plt.plot(xs, val['costs'][inds], label=key)
        if given_time:
            plt.xlabel('Temps de calcul (en sec.)')
        else:
            plt.xlabel('Iteration')
        plt.ylabel('Coût')
        if logscale: plt.loglog()
        plt.legend()
        plt.grid(True)
        if show: plt.show()


if __name__ == "__main__":
    import ordinary_least_square as ols

    X = np.matrix([[1, 2], [3, 4], [1, 4]])
    Y = np.matrix([[0.1], [0.9], [0.5]])

    manager = Manager(X, Y, ols.cost)

    # la recherche est aléatoire, regardons deux runs
    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire", 100, 10 ** (-5))
    manager.run(ols.RandomSearch(X, Y), "recherche aléatoire (bis)", 100,
                10 ** (-5))

    manager.show_plots()
