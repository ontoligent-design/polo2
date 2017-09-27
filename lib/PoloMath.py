import scipy as sp
import math

def JSdivergence(p1,p2):
    P1 = p1/np.sum(p1)
    P2 = p2/np.sum(p2)
    M = .5 * (P1+P2)
    return .5 * (sp.stats.entropy(P1,M) + sp.stats.entropy(P2,M))

def get_mutual_info(p_iAj, p_i, p_j):
    if p_iAj == 0:
        return 0
    x = p_iAj / (p_i * p_j)
    i_ij = 0
    try:
        i_ij = p_iAj * math.log(x)
    except ValueError:
        print("Log bogged at {} ({}) :-(".format(x,round(x,4)))
    finally:
        return i_ij
