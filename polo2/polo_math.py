import numpy as np
import scipy.stats as sps
from scipy.spatial import distance 
# from sklearn.metrics.pairwise import distance_metrics
import math

class PoloMath():

    @staticmethod
    def entropy(s1):
        return sps.entropy(s1, base=2)

    @staticmethod
    def cosine_sim(x, y):
        """Cosine similarity between two vectors"""
        return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
        # return 1 - cosine(x, y)

    @staticmethod
    def cosine_dist(x, y):
        """Cosine distance between two vectors. 1 - cosine similarity."""
        return distance.cosine(x, y)
    
    @staticmethod
    def js_divergence(p1, p2):
        """Jennsen-Shannon Divergence."""
        p = sps.entropy(p1, p2, 2)
        q = sps.entropy(p2, p1, 2)
        jsd = (p + q) / 2
        return jsd

    @staticmethod
    def js_dist(x, y):
        """Jennsen-Shannon Distance. Square root of the convergence."""  
        return distance.jensenshannon(x, y)

    @staticmethod
    def jaccard_dist(x, y):
        """Jaccard Distance. Square root of the convergence."""  
        return distance.jaccard(x, y)

    @staticmethod
    def euclidean_dist(x, y):
        """Simple Euclidean distance"""
        # return math.sqrt(((s1 - s2)**2).sum())
        return distance.euclidean(x, y)

    @staticmethod
    def standard_euclidean_dist(x, y, V):
        return distance.seuclidean(x, y, V)

    @staticmethod
    def chebyshev_dist(x, y):
        return distance.chebyshev(x, y)

    @staticmethod
    def manhattan_dist(x, y):
        return distance.cityblock(x, y)

    @staticmethod
    def pwmi(p_a, p_b, p_ab, norm=.000001):
        """Computes the adjusted point-wise mutual information of two items (a and b)
        that appear in container vectors of some kind, e.g. items in a shopping
        basket."""
        # if p_ab == 0: p_ab = .000001  # To prevent craziness in prob calcs
        p_ab += norm
        i_ab = math.log2(p_ab / (p_a * p_b))  # Raw
        try:
            i_ab = i_ab / (math.log2(p_ab) * -1)  # Adjusted
        except ZeroDivisionError:
            i_ab = 0
        return i_ab

    @staticmethod
    def jscore(s1, s2, thresh = 0):
        """Computes the Jaccard score (aka distance) for two vectors (series). Series passed must
        share an index. This condition will be met for an unstacked matrix of weights or counts,
        where the two series belong to the matrix."""
        A = set(s1[s1 > thresh].index)
        B = set(s2[s2 > thresh].index)
        if len(A | B) > 0:
            return 1 - (len(A & B) / len(A | B))
        else:
            return -1  # Is this correct?

    @staticmethod
    def kl_distance(s1, s2):
        """Kullback-Leibler distance"""
        return sps.entropy(s1, s2, 2)

    @staticmethod
    def softmax(x):
        """Computes softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
