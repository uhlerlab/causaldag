import itertools as itr
from causaldag.classes import UndirectedGraph
from causaldag.utils.ci_tests import CI_Tester, gauss_ci_test
from numpy import sqrt, log1p, ndenumerate, errstate, diagonal, fill_diagonal
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


def partial_correlation_threshold(precision, n=None, alpha=None):
    if n is None:
        return precision

    assert(len(precision.shape) == 2)
    r = precision/sqrt(diagonal(precision))/sqrt(diagonal(precision))[:, None]
    p = r.shape[0]
    n_cond = p - 2

    # note: log1p(2r/(1-r)) = log((1+r)/(1-r)) but is more numerically stable for r near 0
    # r = 1 causes warnings but gives the correct answer
    with errstate(divide='ignore', invalid='ignore'):
        statistic = sqrt(n - n_cond - 3) * abs(.5 * log1p(2*r/(1 - r)))

    p_values = 1 - .5*(1 + erf(statistic/sqrt(2)))

    zero_ixs = p_values > alpha
    fill_diagonal(zero_ixs, False)
    r[zero_ixs] = 0

    return r * sqrt(diagonal(precision))*sqrt(diagonal(precision))[:, None]


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
    r = partial_correlation_threshold(ci_tester.suffstat["P"], ci_tester.suffstat['n'], ci_tester.kwargs.get('alpha'))
    edges = {(i, j) for (i, j), val in ndenumerate(r) if val != 0 and i != j}

    return UndirectedGraph(set(range(ci_tester.suffstat["P"].shape[0])), edges)


