import itertools as itr
from ...classes import UndirectedGraph
from ...utils.ci_tests import CI_Tester, gauss_ci_test
from numpy import sqrt, log1p, ndenumerate, errstate
from scipy.special import erf


def threshold_ug(nodes: set, ci_tester: CI_Tester) -> UndirectedGraph:
    """
    Estimate an undirected graph by testing whether each pair of nodes is independent given all others.

    Parameters
    ----------
    nodes:
        Nodes in the graph.
    ci_tester:
        Conditional independence tester.

    Examples
    --------
    TODO
    """
    if hasattr(ci_tester, 'ci_test') and ci_tester.ci_test == gauss_ci_test:
        return threshold_ug_gauss(ci_tester)
    edges = {(i, j) for i, j in itr.combinations(nodes, 2) if not ci_tester.is_ci(i, j, nodes - {i, j})}
    return UndirectedGraph(nodes, edges)


def threshold_ug_gauss(ci_tester):
    """
    Estimate an undirected graph by testing whether each pair of nodes is independent given all others,
    which reduces to thresholding partial correlations (after the Fisher z-transform) for multivariate Gaussian
    data.

    Parameters
    ----------
    ci_tester:
        Conditional independence tester.

    Examples
    --------
    TODO
    """
    rho = ci_tester.suffstat.get('rho')
    if rho is not None:
        r = rho
    else:
        raise ValueError

    n = ci_tester.suffstat['n']
    p = ci_tester.suffstat['C'].shape[0]
    n_cond = p-2

    # note: log1p(2r/(1-r)) = log((1+r)/(1-r)) but is more numerically stable for r near 0
    # r = 1 causes warnings but gives the correct answer
    with errstate(divide='ignore', invalid='ignore'):
        statistic = sqrt(n - n_cond - 3) * abs(.5 * log1p(2*r/(1 - r)))

    p_values = 1 - .5*(1 + erf(statistic/sqrt(2)))
    alpha = ci_tester.kwargs.get('alpha')
    alpha = alpha if alpha is not None else 1e-5
    edges = {(i, j) for (i, j), p in ndenumerate(p_values) if p < alpha if i != j}

    return UndirectedGraph(set(range(p)), edges)


