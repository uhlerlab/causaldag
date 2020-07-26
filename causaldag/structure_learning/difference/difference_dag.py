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
    [2] Meinshausen, N. and Buhlmann, P. (2010). Stability selection.
    Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), pp.417-473.
"""

from causaldag.structure_learning.difference.difference_ug import dci_undirected_graph
from causaldag.utils.ci_tests import gauss_ci_suffstat
from causaldag.utils.core_utils import powerset
from causaldag.utils.regression import RegressionHelper
from scipy.special import ncfdtr
from numpy.linalg import inv
import numpy as np
import itertools
from joblib import Parallel, delayed
from sklearn.utils import safe_mask
from sklearn.utils.random import sample_without_replacement
import networkx as nx


def dci(
        X1,
        X2,
        alpha_ug: float = 1.0,
        alpha_skeleton: float = 0.1,
        alpha_orient: float = 0.1,
        max_set_size: int = 3,
        difference_ug: list = None,
        nodes_cond_set: set = None,
        max_iter: int = 1000,
        edge_threshold: float = 0,
        verbose: int = 0,
        lam: float = 0
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
    max_set_size: int, default = 3
        Maximum conditioning set size used to test regression invariance.
        Smaller maximum conditioning set size results in faster computation time. For large datasets recommended max_set_size is 3.
        If None, conditioning sets of all sizes will be used.
    difference_ug: list, default = None
        List of tuples that represents edges in the difference undirected graph. If difference_ug is None, 
        KLIEP algorithm for estimating the difference undirected graph will be run. 
        If the number of nodes is small, difference_ug could be taken to be the complete graph between all the nodes.
    nodes_cond_set: set
        Nodes to be considered as conditioning sets.
    max_iter: int, default = 1000
        Maximum number of iterations for gradient descent in KLIEP algorithm.
    edge_threshold: float, default = 0
        Edge weight cutoff for keeping an edge for KLIEP algorithm (all edges above or equal to this threshold are kept).
    verbose: int, default = 0
        The verbosity level of logging messages.
    lam: float, default = 0
        Amount of regularization for regression (becomes ridge regression if nonzero).

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

    References
    ----------
        [1] Wang, Y., Squires, C., Belyaeva, A., & Uhler, C. (2018). Direct estimation of differences in causal graphs. 
        In Advances in Neural Information Processing Systems (pp. 3770-3781).
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
    if difference_ug is None or nodes_cond_set is None:
        difference_ug, nodes_cond_set = dci_undirected_graph(X1, X2, alpha=alpha_ug, max_iter=max_iter, edge_threshold=edge_threshold,
                                             verbose=verbose)
    
    # estimate the skeleton of the difference-DAG 
    skeleton = dci_skeleton(X1, X2, difference_ug, nodes_cond_set, rh1=rh1, rh2=rh2, alpha=alpha_skeleton, max_set_size=max_set_size, verbose=verbose, lam=lam)
    # orient edges of the skeleton of the difference-DAG
    adjacency_matrix = dci_orient(X1, X2, skeleton, nodes_cond_set, rh1=rh1, rh2=rh2, alpha=alpha_orient, max_set_size=max_set_size,
                                                  verbose=verbose)
    return adjacency_matrix


def dci_stability_selection(
        X1,
        X2,
        alpha_ug_grid: list = [0.1, 1, 10],
        alpha_skeleton_grid: list = [0.1, 0.5],
        alpha_orient_grid: list = [0.001, 0.1],
        max_set_size: int = 3,
        difference_ug: list = None,
        nodes_cond_set: set = None,
        max_iter: int = 1000,
        edge_threshold: float = 0.05,
        sample_fraction: float = 0.7,
        n_bootstrap_iterations: int = 50,
        bootstrap_threshold: float = 0.5,
        n_jobs: int = 1,
        random_state: int = None,
        verbose: int = 0,
        lam: float = 0
):
    """
    Runs Difference Causal Inference (DCI) algorithm with stability selection to estimate the difference-DAG between two settings. 
    Bootstrap samples are generated from two input datasets and DCI is run across bootstrap samples and across different 
    combinations of hyperparameters. Edges that reliably appear across different runs are considered stable and are output as the difference-DAG.

    Parameters
    ----------
    X1: array, shape = [n_samples, n_features]
        First dataset.    
    X2: array, shape = [n_samples, n_features]
        Second dataset.
    alpha_ug_grid: array-like, default = [0.1, 1, 10]
        Grid of values to iterate over representing L1 regularization parameter for estimating the difference undirected graph via KLIEP algorithm.
    alpha_skeleton_grid: array-like, default = [0.1, 0.5]
        Grid of values to iterate over representing significance level parameter for determining presence of edges in the skeleton of the difference graph. 
        Lower alpha_skeleton results in sparser difference graph.
    alpha_orient_grid: array-like, default = [0.001, 0.1]
        Grid of values to iterate over representing significance level parameter for determining orientation of an edge. 
        Lower alpha_orient results in more directed edges in the difference-DAG.
    max_set_size: int, default = 3
        Maximum conditioning set size used to test regression invariance.
        Smaller maximum conditioning set size results in faster computation time. For large datasets recommended max_set_size is 3.
    difference_ug: list, default = None
        List of tuples that represents edges in the difference undirected graph. If difference_ug is None, 
        KLIEP algorithm for estimating the difference undirected graph will be run. 
        If the number of nodes is small, difference_ug could be taken to be the complete graph between all the nodes.
    nodes_cond_set: set
        Nodes to be considered as conditioning sets.
    max_iter: int, default = 1000
        Maximum number of iterations for gradient descent in KLIEP algorithm.
    edge_threshold: float, default = 0.05
        Edge weight cutoff for keeping an edge for KLIEP algorithm (all edges above or equal to this threshold are kept).
    sample_fraction: float, default = 0.7
        The fraction of samples to be used in each bootstrap sample.
        Should be between 0 and 1. If 1, all samples are used.
    n_bootstrap_iterations: int, default = 50
        Number of bootstrap samples to create.
    bootstrap_threshold: float, default = 0.5
        Threshold defining the minimum cutoff value for the stability scores. Edges with stability scores above
        the bootstrap_threshold are kept as part of the difference-DAG.
    n_jobs: int, default = 1
        Number of jobs to run in parallel.
    random_state: int, default = None
        Seed used by the random number generator.
    verbose: int, default = 0
        The verbosity level of logging messages.
    lam: float, default = 0
        Amount of regularization for regression (becomes ridge regression if nonzero).

    See Also
    --------
    dci, dci_undirected_graph, dci_skeleton, dci_orient

    Returns
    -------
    adjacency_matrix: array, shape  = [n_features, n_features]
        Estimated difference-DAG. Edges that were found to be different between two settings but the orientation
        could not be determined, are represented by assigning 1 in both directions, i.e. adjacency_matrix[i,j] = 1
        and adjacency_matrix[j,i] = 1. Otherwise for oriented edges, only adjacency_matrix[i,j] = 1 is assigned. 
        Assignment of 0 in the adjacency matrix represents no edge.
    stability_scores: array, shape = [n_params, n_features, n_features]
        Stability score of each edge for for each combination of hyperparameters.

    References
    ----------
        [1] Wang, Y., Squires, C., Belyaeva, A., & Uhler, C. (2018). Direct estimation of differences in causal graphs. 
        In Advances in Neural Information Processing Systems (pp. 3770-3781).
        [2] Meinshausen, N. and Buhlmann, P. (2010). Stability selection.
           Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), pp.417-473.
    """

    _, n_variables = X1.shape
    n_params = len(alpha_ug_grid) * len(alpha_skeleton_grid) * len(alpha_orient_grid)

    hyperparams = itertools.product(alpha_ug_grid, alpha_skeleton_grid, alpha_orient_grid)
    stability_scores = np.zeros((n_params, n_variables, n_variables))

    for idx, params in enumerate(hyperparams):
        if verbose > 0:
            print(
                "Fitting estimator for alpha_ug = %.5f, alpha_skeleton = %.5f, alpha_orient = %.5f with %d bootstrap iterations" %
                (params[0], params[1], params[2], n_bootstrap_iterations))

        bootstrap_samples1 = bootstrap_generator(n_bootstrap_iterations, sample_fraction,
                                                 X1, random_state=random_state)
        bootstrap_samples2 = bootstrap_generator(n_bootstrap_iterations, sample_fraction,
                                                 X2, random_state=random_state)

        bootstrap_results = Parallel(n_jobs, verbose=verbose
                                     )(delayed(dci)(X1[safe_mask(X1, subsample1), :],
                                                    X2[safe_mask(X2, subsample2), :],
                                                    alpha_ug=params[0],
                                                    alpha_skeleton=params[1],
                                                    alpha_orient=params[2],
                                                    max_set_size=max_set_size,
                                                    difference_ug=difference_ug,
                                                    nodes_cond_set=nodes_cond_set,
                                                    max_iter=max_iter,
                                                    edge_threshold=edge_threshold,
                                                    verbose=verbose,
                                                    lam=lam)
                                       for subsample1, subsample2 in zip(bootstrap_samples1, bootstrap_samples2))

        stability_scores[idx] = np.array(bootstrap_results).mean(axis=0)

    adjacency_matrix = choose_stable_variables(stability_scores, bootstrap_threshold=bootstrap_threshold)
    return adjacency_matrix, stability_scores


def choose_stable_variables(stability_scores, bootstrap_threshold=0.5):
    """Returns adjacency matrix corresponding to edges with stability scores above threshold."""
    return (stability_scores.max(axis=0) > bootstrap_threshold).astype('float')


def bootstrap_generator(n_bootstrap_iterations, sample_fraction, X, random_state=None):
    """Generates bootstrap samples from dataset."""
    n_samples = len(X)
    n_subsamples = np.floor(sample_fraction * n_samples).astype(int)
    for _ in range(n_bootstrap_iterations):
        subsample = sample_without_replacement(n_samples, n_subsamples, random_state=random_state)
        yield subsample


def dci_skeleton(
        X1,
        X2,
        difference_ug: list,
        nodes_cond_set: set,
        rh1: RegressionHelper = None,
        rh2: RegressionHelper = None,
        alpha: float = 0.1,
        max_set_size: int = 3,
        verbose: int = 0,
        lam: float = 0
):
    """
    Estimates the skeleton of the difference-DAG.

    Parameters
    ----------
    X1: array, shape = [n_samples, n_features]
        First dataset.    
    X2: array, shape = [n_samples, n_features]
        Second dataset.
    difference_ug: list
        List of tuples that represents edges in the difference undirected graph.
    nodes_cond_set: set
        Nodes to be considered as conditioning sets.
    rh1: RegressionHelper, default = None
        Sufficient statistics estimated based on samples in the first dataset, stored in RegressionHelper class.
    rh2: RegressionHelper, default = None
        Sufficient statistics estimated based on samples in the second dataset, stored in RegressionHelper class.
    alpha: float, default = 0.1
        Significance level parameter for determining presence of edges in the skeleton of the difference graph.
        Lower alpha results in sparser difference graph.
    max_set_size: int, default = 3
        Maximum conditioning set size used to test regression invariance.
        Smaller maximum conditioning set size results in faster computation time. For large datasets recommended max_set_size is 3.
    verbose: int, default = 0
        The verbosity level of logging messages.
    lam: float, default = 0
        Amount of regularization for regression (becomes ridge regression if nonzero).

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

    if rh1 is None or rh2 is None:
        # obtain sufficient statistics
        suffstat1 = gauss_ci_suffstat(X1)
        suffstat2 = gauss_ci_suffstat(X2)
        rh1 = RegressionHelper(suffstat1)
        rh2 = RegressionHelper(suffstat2)

    n1 = rh1.suffstat['n']
    n2 = rh2.suffstat['n']

    skeleton = {(i, j) for i, j in difference_ug}

    for i, j in difference_ug:
        for cond_set in powerset(nodes_cond_set - {i, j}, r_max=max_set_size):
            cond_set_i, cond_set_j = [*cond_set, j], [*cond_set, i]

            # calculate regression coefficients (j regressed on cond_set_j) for both datasets
            beta1_i, var1_i, precision1 = rh1.regression(i, cond_set_i, lam=lam)
            beta2_i, var2_i, precision2 = rh2.regression(i, cond_set_i, lam=lam)

            # compute statistic and p-value
            j_ix = cond_set_i.index(j)
            stat_i = (beta1_i[j_ix] - beta2_i[j_ix]) ** 2 * \
                 inv(var1_i * precision1 / (n1 - 1) + var2_i * precision2 / (n2 - 1))[j_ix, j_ix]
            pval_i = 1 - ncfdtr(1, n1 + n2 - len(cond_set_i) - len(cond_set_j), 0, stat_i)

            #  remove i-j from skeleton if i regressed on (j, cond_set) is invariant
            i_invariant = pval_i > alpha
            if i_invariant:
                if verbose > 0:
                    print("Removing edge %d->%d since p-value=%.5f > alpha=%.5f with cond set %s" % (i, j, pval_i, alpha, cond_set_i))
                skeleton.remove((i, j))
                break
            elif verbose > 0: print(f"Keeping edge %d->%d for now, since p-value=%.5f < alpha=%.5f with cond set %s" % (i, j, pval_i, alpha, cond_set_i))

            # calculate regression coefficients (i regressed on cond_set_i) for both datasets
            beta1_j, var1_j, precision1 = rh1.regression(j, cond_set_j)
            beta2_j, var2_j, precision2 = rh2.regression(j, cond_set_j)

            # compute statistic and p-value
            i_ix = cond_set_j.index(i)
            stat_j = (beta1_j[i_ix] - beta2_j[i_ix]) ** 2 * \
                 inv(var1_j * precision1 / (n1 - 1) + var2_j * precision2 / (n2 - 1))[i_ix, i_ix]
            pval_j = 1 - ncfdtr(1, n1 + n2 - len(cond_set_i) - len(cond_set_j), 0, stat_j)

            #  remove i-j from skeleton if j regressed on (i, cond_set) is invariant
            j_invariant = pval_j > alpha
            if j_invariant:
                if verbose > 0:
                    print("Removing edge %d->%d since p-value=%.5f > alpha=%.5f with cond set %s" % (i, j, pval_j, alpha, cond_set_j))
                skeleton.remove((i, j))
                break
            elif verbose > 0: print(f"Keeping edge %d->%d for now, since p-value=%.5f < alpha=%.5f with cond set %s" % (i, j, pval_j, alpha, cond_set_j))

    return skeleton


def dci_orient(
        X1, 
        X2,
        skeleton: set,
        nodes_cond_set: set,
        rh1: RegressionHelper = None,
        rh2: RegressionHelper = None,
        alpha: float = 0.1,
        max_set_size: int = 3,
        verbose: int = 0
):
    """
    Orients edges in the skeleton of the difference DAG.

    Parameters
    ----------
    X1: array, shape = [n_samples, n_features]
        First dataset.    
    X2: array, shape = [n_samples, n_features]
        Second dataset.
    skeleton: set
        Set of edges in the skeleton of the difference-DAG.
    nodes_cond_set: set
        Nodes to be considered as conditioning sets.
    rh1: RegressionHelper, default = None
        Sufficient statistics estimated based on samples in the first dataset, stored in RegressionHelper class.
    rh2: RegressionHelper, default = None
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

    if rh1 is None or rh2 is None:
        # obtain sufficient statistics
        suffstat1 = gauss_ci_suffstat(X1)
        suffstat2 = gauss_ci_suffstat(X2)
        rh1 = RegressionHelper(suffstat1)
        rh2 = RegressionHelper(suffstat2)

    nodes = {i for i, j in skeleton} | {j for i, j in skeleton}
    oriented_edges = set()

    n1 = rh1.suffstat['n']
    n2 = rh2.suffstat['n']
    for i, j in skeleton:
        for cond_i, cond_j in zip(powerset(nodes_cond_set - {i}, r_max=max_set_size), powerset(nodes_cond_set - {j}, r_max=max_set_size)):
            # compute residual variances for i
            beta1_i, var1_i, _ = rh1.regression(i, list(cond_i))
            beta2_i, var2_i, _ = rh2.regression(i, list(cond_i))
            # compute p-value for invariance of residual variances for i
            pvalue_i = ncfdtr(n1 - len(cond_i), n2 - len(cond_i), 0, var1_i / var2_i)
            pvalue_i = 2 * min(pvalue_i, 1 - pvalue_i)

            # compute residual variances for j
            beta1_j, var1_j, _ = rh1.regression(j, list(cond_j))
            beta2_j, var2_j, _ = rh2.regression(j, list(cond_j))
            # compute p-value for invariance of residual variances for j
            pvalue_j = ncfdtr(n1 - len(cond_j), n2 - len(cond_j), 0, var1_j / var2_j)
            pvalue_j = 2 * min(pvalue_j, 1 - pvalue_j)

            if ((pvalue_i > alpha) | (pvalue_j > alpha)):
                # orient the edge according to highest p-value
                if pvalue_i > pvalue_j:
                    edge = (j, i) if j in cond_i else (i, j)
                    pvalue_used = pvalue_i
                else:
                    edge = (i, j) if i in cond_j else (j, i)
                    pvalue_used = pvalue_j
                oriented_edges.add(edge)

                if verbose > 0:
                    print("Oriented (%d, %d) as %s since p-value=%.5f > alpha=%.5f" % (i, j, edge, pvalue_used, alpha))
                break

    # orient edges via graph traversal
    unoriented_edges_before_traversal = skeleton - oriented_edges - {(j, i) for i, j in oriented_edges}
    unoriented_edges = unoriented_edges_before_traversal.copy()
    g = nx.DiGraph()
    for i, j in oriented_edges:
        g.add_edge(i, j)
    g.add_nodes_from(nodes)


    for i, j in unoriented_edges_before_traversal:
        chain_path = list(nx.all_simple_paths(g, source=i, target=j))
        if len(chain_path) > 0:
            oriented_edges.add((i, j))
            unoriented_edges.remove((i, j))
            if verbose > 0:
                print("Oriented (%d, %d) as %s with graph traversal" % (i, j, (i, j)))
        else:
            chain_path = list(nx.all_simple_paths(g, source=j, target=i))
            if len(chain_path) > 0:
                oriented_edges.add((j, i))
                unoriented_edges.remove((i, j))
                if verbose > 0:
                    print("Oriented (%d, %d) as %s with graph traversal" % (i, j, (j, i)))

    # form an adjacency matrix containing directed and undirected edges
    num_nodes = X1.shape[1]
    adjacency_matrix = edges2adjacency(num_nodes, unoriented_edges, undirected=True) + edges2adjacency(num_nodes,
                                                                                                       oriented_edges,
                                                                                                       undirected=False)
    return adjacency_matrix


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


def get_directed_and_undirected_edges(adjacency_matrix):
    """
    Given an adjacency matrix, which contains both directed and undirected edges,
    this function returns two adjancy matrices containing directed and undirected edges separately.
    Useful for plotting the difference causal graph.

    Parameters
    ----------
    adjacency_matrix: array, shape  = [num_nodes, num_nodes]
        Adjacency matrix representing partially directed acyclic graph,
        which containts both undirected and directed edges. 
        Each entry should be either 0 or 1, representing absence or presence of an edge, respectively.

    Returns
    -------
    adjacency_matrix_directed: array, shape  = [num_nodes, num_nodes]
        Adjacency matrix containing only directed edges.
    adjacency_matrix_undirected: array, shape  = [num_nodes, num_nodes]
        Adjacency matrix containing only undirected edges.
    """

    adjacency_matrix = adjacency_matrix.astype('float')
    adjacency_matrix_sym = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix_undirected = (adjacency_matrix_sym == 2).astype('float')
    adjacency_matrix_directed = (adjacency_matrix_sym == 1).astype('float')
    adjacency_matrix_directed[adjacency_matrix_directed == 1] = adjacency_matrix[adjacency_matrix_directed==1]
    return adjacency_matrix_directed, adjacency_matrix_undirected
