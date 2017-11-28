import numpy as np
import scipy.stats as sps
import math


class PoloMath():

    @staticmethod
    def cosine_sim(x, y):
        """ x and y are two comparable distribution vectors, e.g. words for a topic"""
        c1 = math.sqrt(sum([m * m for m in x]))
        c2 = math.sqrt(sum([m * m for m in y]))
        c3 = math.sqrt(sum([m * n for m, n in zip(x, y)]))
        try:
            c4 = c3 / (c1 * c2)
        except:
            c4 = None
        return c4

    @staticmethod
    def js_divergence(p1, p2):
        """Computes the Jensen=Shannon Divergence between two vectors (series)."""
        P1 = p1 / np.sum(p1)
        P2 = p2 / np.sum(p2)
        M = .5 * (P1 + P2)
        return .5 * (sps.entropy(P1, M) + sps.entropy(P2, M))

    @staticmethod
    def pwmi(p_a, p_b, p_ab):
        """Computes the adjusted point-wise mutual information of two items (a and b)
        that appear in container vectors of some kind, e.g. items in a shopping
        basket."""
        if p_ab > 0:
            i_ab = math.log(p_ab / (p_a * p_b))  # Raw
            i_ab = i_ab / (math.log2(p_ab) * -1) # Adjusted
        else:
            i_ab = None
        return i_ab

    @staticmethod
    def jscore(s1, s2, thresh = 0):
        """Computes teh Jaccard score (aka distance) for two vectors (series). Series passed must
        share an index. This condition will be met for an unstacked matrix of weights or counts,
        where the two series belong to the matrix."""
        A = set(s1[s1 > thresh].index)
        B = set(s2[s2 > thresh].index)
        return 1 - (len(A & B) / len(A | B))
