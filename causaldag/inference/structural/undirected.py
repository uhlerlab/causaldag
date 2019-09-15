import itertools as itr
from ...classes import UndirectedGraph
from ...utils.ci_tests import CI_Tester, gauss_ci_test
from numpy import sqrt, log1p
from scipy.special import erf
import numpy as np


def threshold_ug(nodes: set, ci_tester: CI_Tester) -> UndirectedGraph:
    if ci_tester.ci_test == gauss_ci_test:
        return threshold_ug_gauss(ci_tester)
    edges = {(i, j) for i, j in itr.combinations(nodes, 2) if not ci_tester.is_ci(i, j, nodes - {i, j})}
    return UndirectedGraph(nodes, edges)


def threshold_ug_gauss(ci_tester):
    rho = ci_tester.suffstat.get('rho')
    if rho is not None:
        r = rho
    else:
        raise ValueError

    n = ci_tester.suffstat['n']
    p = ci_tester.suffstat['C'].shape[0]
    n_cond = p-2
    statistic = sqrt(n - n_cond - 3) * abs(.5 * log1p(2*r/(1 - r)))
    p_values = 1 - .5*(1 + erf(statistic/sqrt(2)))
    alpha = ci_tester.kwargs.get('alpha')
    alpha = alpha if alpha is not None else 1e-5
    edges = {(i, j) for (i, j), p in np.ndenumerate(p_values) if p < alpha if i != j}

    return UndirectedGraph(set(range(p)), edges)


