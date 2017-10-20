import numpy as np
import scipy.stats as sps
import math

class PoloMath():

    @staticmethod
    def JSdivergence(p1, p2):
        P1 = p1/np.sum(p1)
        P2 = p2/np.sum(p2)
        M = .5 * (P1 + P2)
        return .5 * (sps.entropy(P1, M) + sps.entropy(P2, M))

    @staticmethod
    def pwmi(p_a, p_b, p_ab):
        if p_ab > 0:
            i_ab = math.log(p_ab / (p_a * p_b))
            i_ab = i_ab / (math.log2(p_ab) * -1)
        else:
            i_ab = None
        return i_ab


