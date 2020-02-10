"""
===============================
Estimation of differences between directed graphs given two datasets representing two settings.
===============================
This module contains functions for estimating the differences between two causal 
directed acyclic graph (DAG) models given samples from each model.

References
----------
   [1] Wang, Y., Squires, C., Belyaeva, A., & Uhler, C. (2018). Direct estimation of differences in causal graphs. 
   In Advances in Neural Information Processing Systems (pp. 3770-3781).
"""


from causaldag.structure_learning.difference.difference_ug import dci_undirected_graph
from causaldag.utils.ci_tests import gauss_ci_suffstat
from causaldag.utils.core_utils import powerset
from causaldag.utils.regression import RegressionHelper
from scipy.special import ncfdtr
from numpy.linalg import inv


def dci(
        X1,
        X2,
        alpha_ug: float=1.0,
        alpha_skeleton: float=0.1,
        alpha_orient: float=0.1,
        max_set_size: int=None,
        difference_ug: list=None,
        max_iter: int=1000,
        edge_threshold: float=0.05,
        verbose: int=0
):
    """
    Uses the Difference Causal Inference (DCI) algorithm to estimate the difference-DAG between two settings.

    Parameters
    ----------
    X1: array, shape = [n_samples, n_features]
        First dataset.    
    X2: array, shape = [n_samples, n_features]
        Second dataset.
    alpha_ug: float, default = 1.0
        L1 regularization parameter for estimating the difference undirected graph via KLIEP algorithm.
    alpha_skeleton: float, default = 0.1
        Significance level parameter for determining presence of edges in the skeleton of the difference graph. 
        Lower alpha_skeleton results in sparser difference graph.
    alpha_orient: float, default = 0.1
        Significance level parameter for determining orientation of an edge. 
        Lower alpha_orient results in more directed edges in the difference-DAG.
    max_set_size: int, default = None
        Maximum conditioning set size used to test regression invariance.
        Smaller maximum conditioning set size results in faster computation time. For large datasets recommended max_set_size is 3.
    difference_ug: list, default = None
        List of tuples that represents edges in the difference undirected graph. If difference_ug is None, 
        KLIEP algorithm for estimating the difference undirected graph will be run. 
        If the number of nodes is small, difference_ug could be taken to be the complete graph between all the nodes.
    max_iter: int, default = 1000
        Maximum number of iterations for gradient descent in KLIEP algorithm.
    edge_threshold: float, default = 0.05
        Edge weight cutoff for keeping an edge for KLIEP algorithm (all edges above or equal to this threshold are kept).
    verbose: int, default = 0
        The verbosity level of logging messages.

    See Also
    --------
    dci_undirected_graph, dci_skeleton, dci_orient

    Returns
    -------
    adjacency_matrix: array, shape  = [n_features, n_features]
        Estimated difference-DAG. Edges that were found to be different between two settings but the orientation
        could not be determined, are represented by assigning 1 in both directions, i.e. adjacency_matrix[i,j] = 1
        and adjacency_matrix[j,i] = 1. Otherwise for oriented edges, only adjacency_matrix[i,j] = 1 is assigned. 
        Assignment of 0 in the adjacency matrix represents no edge.
    """

    assert 0 <= alpha_skeleton <= 1, "alpha_skeleton must be in [0,1] range."
    assert 0 <= alpha_orient <= 1, "alpha_orient must be in [0,1] range."

    num_nodes = X1.shape[1]
    # obtain sufficient statistics
    suffstat1 = gauss_ci_suffstat(X1)
    suffstat2 = gauss_ci_suffstat(X2)    
    rh1 = RegressionHelper(suffstat1)
    rh2 = RegressionHelper(suffstat2)
    
    # compute the difference undirected graph via KLIEP if the differece_ug is not provided
    if difference_ug is None:
        difference_ug = dci_undirected_graph(X1, X2, alpha=alpha_ug, max_iter=max_iter, edge_threshold=edge_threshold, verbose=verbose)
    
    # estimate the skeleton of the difference-DAG 
    skeleton = dci_skeleton(difference_ug, rh1, rh2, alpha=alpha_skeleton, max_set_size=max_set_size, verbose=verbose)
    # orient edges of the skeleton of the difference-DAG
    edges_oriented, edges_unoriented = dci_orient(skeleton, rh1, rh2, alpha=alpha_orient, max_set_size=max_set_size, verbose=verbose)

    adjacency_matrix = edges2adjacency(num_nodes, edges_unoriented, undirected=True) + edges2adjacency(num_nodes, edges_oriented, undirected=False)
    return adjacency_matrix


def dci_skeleton(
        difference_ug: list,
        rh1: RegressionHelper,
        rh2: RegressionHelper,
        alpha: float=0.1,
        max_set_size: int=3,
        verbose: int=0
):
    """
    Estimates the skeleton of the difference-DAG.

    Parameters
    ----------
    difference_ug: list
        List of tuples that represents edges in the difference undirected graph.
    rh1: RegressionHelper
        Sufficient statistics estimated based on samples in the first dataset, stored in RegressionHelper class.
    rh2: RegressionHelper
        Sufficient statistics estimated based on samples in the second dataset, stored in RegressionHelper class.
    alpha: float, default = 0.1
        Significance level parameter for determining presence of edges in the skeleton of the difference graph.
        Lower alpha results in sparser difference graph.
    max_set_size: int, default = None
        Maximum conditioning set size used to test regression invariance.
        Smaller maximum conditioning set size results in faster computation time. For large datasets recommended max_set_size is 3.
    verbose: int, default = 0
        The verbosity level of logging messages.

    See Also
    --------
    dci, dci_undirected_graph, dci_orient

    Returns
    -------
    skeleton: set
        Set of edges in the skeleton of the difference-DAG.
    """

    if verbose > 0:
        print("DCI skeleton estimation...")

    assert 0 <= alpha <= 1, "alpha must be in [0,1] range."

    n1 = rh1.suffstat['n']
    n2 = rh2.suffstat['n']
    nodes = get_nodes_in_graph(difference_ug)
    skeleton = {frozenset({i, j}) for i, j in difference_ug}

    for i, j in difference_ug:
        for cond_set in powerset(nodes - {i, j}, r_max=max_set_size):
            cond_set_i, cond_set_j = [*cond_set, j], [*cond_set, i]

            # calculate regression coefficients (j regressed on cond_set_j) for both datasets
            beta1_i, var1_i, precision1 = rh1.regression(i, cond_set_i)
            beta2_i, var2_i, precision2 = rh2.regression(i, cond_set_i)

            # compute statistic and p-value
            j_ix = cond_set_i.index(j)
            stat_i = (beta1_i[j_ix] - beta2_i[j_ix])**2 * inv(var1_i*precision1/(n1-1) + var2_i*precision2/(n2-1))[j_ix, j_ix]
            pval_i = ncfdtr(1, n1 + n2 - len(cond_set_i) - len(cond_set_j), 0, stat_i)
            pval_i = 2 * min(pval_i, 1 - pval_i)

            #  remove i-j from skeleton if i regressed on (j, cond_set) is invariant
            i_invariant = pval_i > alpha
            if i_invariant:
                if verbose > 0:
                    print("Removing edge %d-%d since p-value=%.5f < alpha=%.5f" %(i, j, pval_i, alpha))
                skeleton.remove(frozenset({i, j}))
                break

            # calculate regression coefficients (i regressed on cond_set_i) for both datasets
            beta1_j, var1_j, precision1 = rh1.regression(j, cond_set_j)
            beta2_j, var2_j, precision2 = rh2.regression(j, cond_set_j)

            # compute statistic and p-value
            i_ix = cond_set_j.index(i)
            stat_j = (beta1_j[i_ix] - beta2_j[i_ix]) ** 2 * inv(var1_j*precision1/(n1-1) + var2_j*precision2/(n2-1))[i_ix, i_ix]
            pval_j = 1 - ncfdtr(1, n1 + n2 - len(cond_set_i) - len(cond_set_j), 0, stat_j)
            pval_j = 2 * min(pval_j, 1 - pval_j)

            #  remove i-j from skeleton if j regressed on (i, cond_set) is invariant
            j_invariant = pval_j > alpha
            if j_invariant:
                if verbose > 0:
                    print("Removing edge %d-%d since p-value=%.5f < alpha=%.5f" %(i, j, pval_j, alpha))
                skeleton.remove(frozenset({i, j}))
                break

    return skeleton


def dci_orient(
        skeleton: set,
        rh1: RegressionHelper,
        rh2: RegressionHelper,
        alpha: float=0.1,
        max_set_size: int=3,
        verbose: int=0
):
    """
    Orients edges in the skeleton of the difference DAG.

    Parameters
    ----------
    skeleton: set
        Set of edges in the skeleton of the difference-DAG.
    rh1: RegressionHelper
        Sufficient statistics estimated based on samples in the first dataset, stored in RegressionHelper class.
    rh2: RegressionHelper
        Sufficient statistics estimated based on samples in the second dataset, stored in RegressionHelper class.
    alpha: float, default = 0.1
        Significance level parameter for determining orientation of an edge.
        Lower alpha results in more directed edges in the difference-DAG.
    max_set_size: int, default = 3
        Maximum conditioning set size used to test regression invariance.
        Smaller maximum conditioning set size results in faster computation time. For large datasets recommended max_set_size is 3.
    verbose: int, default = 0
        The verbosity level of logging messages.

    See Also
    --------
    dci, dci_undirected_graph, dci_skeleton

    Returns
    -------
    oriented_edges: set
        Set of edges in the skeleton of the difference-DAG for which directionality could be determined.
    unoriented_edges: set
        Set of edges in the skeleton of the difference-DAG for which directionality could not be determined.
    """

    if verbose > 0:
        print("DCI edge orientation...")

    assert 0 <= alpha <= 1, "alpha must be in [0,1] range."

    nodes = {i for i, j in skeleton} | {j for i, j in skeleton}
    oriented_edges = set()

    n1 = rh1.suffstat['n']
    n2 = rh2.suffstat['n']
    for i, j in skeleton:
        for cond_i, cond_j in zip(powerset(nodes - {i}, r_max=max_set_size), powerset(nodes - {j}, r_max=max_set_size)):
            # compute residual variances for i
            beta1_i, var1_i, _ = rh1.regression(i, cond_i)
            beta2_i, var2_i, _ = rh2.regression(i, cond_i)
            # compute p-value for invariance of residual variances for i
            pvalue_i = ncfdtr(n1 - len(cond_i), n2 - len(cond_i), 0, var1_i/var2_i)
            pvalue_i = 2 * min(pvalue_i, 1-pvalue_i)

            # compute residual variances for j
            beta1_j, var1_j, _ = rh1.regression(j, cond_j)
            beta2_j, var2_j, _ = rh2.regression(j, cond_j)
            # compute p-value for invariance of residual variances for j
            pvalue_j = ncfdtr(n1 - len(cond_j), n2 - len(cond_j), 0, var1_j / var2_j)
            pvalue_j = 2 * min(pvalue_j, 1 - pvalue_j)

            if ((pvalue_i > alpha) | (pvalue_j > alpha)):
                # orient the edge according to highest p-value
                if pvalue_i > pvalue_j:
                    edge = (j, i) if j in cond_i else (i, j)
                else:
                    edge = (i, j) if i in cond_j else (j, i)
                oriented_edges.add(edge)
                
                if verbose > 0:
                    print("Oriented (%d, %d) as %s" % (i, j, edge))
                break

    unoriented_edges = skeleton - {frozenset({i, j}) for i, j in oriented_edges}
    return oriented_edges, unoriented_edges


def edges2adjacency(num_nodes, edge_set, undirected=False):
    """
    Returns adjacency_matrix given a set of edges. If the edges are considered undirected,
    then the adjacency matrix will be symmetric.

    Parameters
    ----------
    num_nodes: int
        Number of nodes in the graph.
    edge_set: set
        Set of edges in the graph.    
    undirected: bool, default = False
        Whether to consider the edges in the edge set as directed or undirected.

    Returns
    -------
    adjacency_matrix: array, shape  = [num_nodes, num_nodes]
        Adjacency matrix.
    """

    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for parent, child in edge_set:
        adjacency_matrix[parent, child] = 1
        if undirected:
            adjacency_matrix[child, parent] = 1
    return adjacency_matrix


def get_nodes_in_graph(graph):
    """
    Returns nodes that are in the graph.
    """
    return set(np.unique(graph))