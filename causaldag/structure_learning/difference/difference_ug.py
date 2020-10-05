"""
===============================
Estimation of differences between undirected graphs.
===============================
This module contains functions for computing the difference undirected graph given two data sets.
References
----------
.. [1] Liu, S., Quinn, J. A., Gutmann, M. U., Suzuki, T., & Sugiyama, M. (2014). 
   Direct learning of sparse changes in Markov networks by density ratio estimation. 
   Neural computation, 26(6), 1169-1197.
   [2] http://allmodelsarewrong.net/kliep_sparse/demo_sparse.html
"""


import numpy as np 
import scipy
import networkx as nx
from numpy.linalg import pinv
import itertools as itr
from scipy.special import ncfdtr
import ipdb


def dci_undirected_graph(X1, X2, difference_ug_method = 'constraint', alpha=0.01, max_iter=1000, edge_threshold=0, verbose=0):
    """
    Estimates the difference between two undirected graphs directly from two data sets
    using constraint-based or Kullback-Leibler importance estimation procedure (KLIEP).

    Parameters
    ----------
    X1: array, shape = [n_samples, n_features]
        First dataset.    
    X2: array, shape = [n_samples, n_features]
        Second dataset.
    difference_ug_method: str, default = 'constraint'
        Method for computing the undirected difference graph. Must be 'constraint' for constraint-based
        method or 'kliep' for KLIEP.
    alpha: float, default = 1.0
        Parameter for determining the difference undirected graph.
        If difference_ug_method = 'constraint', alpha_ug is the significance level parameter (must be in [0,1] range),
        with higher alpha_ug resulting in more edges in the difference undirected graph.
        If difference_ug_method = 'kliep', alpha_ug is the L1 regularization parameter for estimating 
        the difference undirected graph via KLIEP algorithm.
    max_iter: int, default = 1000
        Maximum number of iterations for gradient descent (KLIEP only).
    edge_threshold: float, default = 0.05
        Edge weight cutoff for keeping an edge where all edges above or equal to this threshold are kept (KLIEP only).
    verbose: int, default = 0
        The verbosity level of logging messages.

    Returns
    -------
    difference_ug: list
        List of tuple of edges in the difference undirected graph.
    nodes_cond_set: set
        Nodes to be considered as conditioning sets.
    """
    if difference_ug_method == 'constraint':
        difference_ug, nodes_cond_set = constraint_diff_ug(X1, X2, alpha=alpha, verbose=verbose)
    elif difference_ug_method == 'kliep':
        difference_ug, nodes_cond_set = kliep_diff_ug(
            X1,
            X2,
            alpha=alpha,
            max_iter=max_iter,
            edge_threshold=edge_threshold,
            verbose=verbose
        )
    else:
        raise ValueError("`difference_ug_method` should be either 'constraint' or 'kliep'")
    return difference_ug, nodes_cond_set


def constraint_diff_ug(X1, X2, alpha=0.01, verbose=0):
    """
    Estimates the difference between two undirected graphs directly from two data sets
    using constraint-based method that relies on comparing precision matrices corresponding
    to each data set.

    Parameters
    ----------
    X1: array, shape = [n_samples, n_features]
        First dataset.    
    X2: array, shape = [n_samples, n_features]
        Second dataset.
    alpha: float, default = 0.01
        Parameter for the constraint-based method, which corresponds to a p-value cutoff
        for hypothesis testing. Higher alpha leads to more edges in the difference undirected graph.
    verbose: int, default = 0
        The verbosity level of logging messages.

    Returns
    -------
    difference_ug: list
        List of tuple of edges in the difference undirected graph.
    nodes_cond_set: set
        Nodes to be considered as conditioning sets.
    """
    assert 0 <= alpha <= 1, "alpha must be in [0,1] range."
    if verbose > 0:
        print("Running constraint-based method to get difference undirected graph...")

    n1, n2, p = X1.shape[0], X2.shape[0], X1.shape[1]
    K1 = pinv(np.cov(X1, rowvar=False))
    K2 = pinv(np.cov(X2, rowvar=False))
    D1 = np.diag(K1)
    D2 = np.diag(K2)
    stats = (K1 - K2)**2 * 1/((np.outer(D1, D1) + K1**2)/n1 + (np.outer(D2, D2) + K2**2)/n2)
    pvals = 1 - ncfdtr(1, n1 + n2 - 2*p + 2, 0, stats)

    diff_ug = {frozenset({i, j}) for i, j in itr.combinations(range(p), 2) if pvals[i, j] <= alpha}
    cond_nodes = {i for i, _ in diff_ug} | {j for _, j in diff_ug}
    
    if verbose > 0:
        print("Difference undirected graph: ", difference_ug)

    return diff_ug, cond_nodes


def kliep_diff_ug(X1, X2, alpha=1.0, max_iter=1000, edge_threshold=0, verbose=0):
    """
    Estimates the difference between two undirected graphs directly from two data sets
    using Kullback-Leibler importance estimation procedure (KLIEP).

    Parameters
    ----------
    X1: array, shape = [n_samples, n_features]
        First dataset.    
    X2: array, shape = [n_samples, n_features]
        Second dataset.
    alpha: float, default = 1.0
        L1 regularization parameter.
    max_iter: int, default = 1000
        Maximum number of iterations for gradient descent.
    edge_threshold: float, default = 0.05
        Edge weight cutoff for keeping an edge (all edges above or equal to this threshold are kept).
    verbose: int, default = 0
        The verbosity level of logging messages.

    Returns
    -------
    difference_ug: list
        List of tuple of edges in the difference undirected graph.
    nodes_cond_set: set
        Nodes to be considered as conditioning sets.
    """

    if verbose > 0:
        print("Running KLIEP to get difference undirected graph...")

    k1 = kernel_linear(X1)
    k2 = kernel_linear(X2)
    theta = naive_subgradient_descent(k1, k2, alpha=alpha, max_iter=max_iter, verbose=verbose)
    difference_ug = compute_difference_graph(X1, theta, edge_threshold=edge_threshold)
    
    # get nodes to be considered in the conditioning sets
    nodes_cond_set = get_nodes_in_graph(difference_ug)
    # remove self-edges from the difference undirected graph
    difference_ug = [tuple((i, j)) for i, j in difference_ug if i != j]
    
    if verbose > 0:
        print("Difference undirected graph: ", difference_ug)

    return difference_ug, nodes_cond_set


def kernel_linear(X):
    """
    Computes polynomial features (order = 2) based on data.
    """
    n, d = X.shape
    kernel_features = np.zeros((n, int((d*(d-1))/2)))
    for i in range(n):
        t = np.matmul(X[i,:].reshape(d,1), X[i,:].reshape(d,1).T)
        kernel_features[i,:] = t[np.triu_indices_from(t,1)]

    kernel_features = np.concatenate((kernel_features, X**2), axis=1)
    return kernel_features


def llkliep(theta, k1, k2):
    """
    Computes the log-likelihood of the model and the gradient.
    """
    loglik = -np.mean(np.matmul(theta.T, k1.T), 1) + scipy.special.logsumexp(np.matmul(theta.T, k2.T), 1)
    log_g_q = np.matmul(theta.T, k2.T) - scipy.special.logsumexp(np.matmul(theta.T, k2.T), 1)
    g_q = np.exp(log_g_q)
    grad = -np.mean(k1, 0).reshape((-1, 1)) + np.matmul(k2.T, g_q.T)
    return loglik[0], grad


def naive_subgradient_descent(k1, k2, alpha=1, max_iter=1000, verbose=0):
    """
    Performs gradient updates to find parameters that maximize the log-likelihood.

    Parameters
    ----------
    k1: array, shape = [n_samples, n_features]
        First dataset after featurization.    
    k2: array, shape = [n_samples, n_features]
        Second dataset after featurization.
    alpha: float, default = 1.0
        L1 regularization parameter.
    max_iter: int, default = 1000
        Maximum number of iterations for gradient descent.
    verbose: int, default = 0
        The verbosity level of logging messages.

    Returns
    -------
    theta: array
        Estimated parameters corresponding to the difference undirected graph.  
    """

    # initialize variables
    d = k1.shape[1]
    theta = np.zeros((d, 1))
    step = 1
    slength = np.inf
    iter = 0
    loglik_old = 1e20

    while (slength > 1e-5) & (iter < max_iter):
        loglik, grad = llkliep(theta, k1, k2)
        g = np.zeros(grad.shape)

        ids = theta.nonzero()
        g[ids] = grad[ids] + alpha*np.sign(theta[ids])

        zero_ids = theta == 0
        ids = zero_ids & (grad > alpha)
        g[ids] = grad[ids] - alpha

        ids = zero_ids & (grad < -alpha)
        g[ids] = grad[ids] + alpha

        # update theta parameters
        theta = theta - step*g/(iter + 1)
        slength = step*np.linalg.norm(g)/(iter + 1)

        loglik_diff = np.abs(loglik - loglik_old)
        loglik_old = loglik
        iter = iter + 1
        if (verbose > 0) & (iter == max_iter):
            print('Maximum iteration reached')
    return theta


def compute_difference_graph(X, theta, edge_threshold=0):
    """
    Obtain difference undirected graph from estimated parameters.
    """
    n, d = X.shape
    delta_ug = np.zeros((d, d))
    delta_ug[np.triu_indices(d, 1)] = theta[0:-d].flatten()
    delta_ug = delta_ug + delta_ug.T
    # set the diagonal
    np.fill_diagonal(delta_ug, theta[-d:])
    # remove edges that are below cutoff threshold
    delta_ug[np.abs(delta_ug) < edge_threshold] = 0
    g = nx.from_numpy_matrix(delta_ug)
    return list(g.edges())


def get_nodes_in_graph(graph):
    """
    Returns nodes that are in the graph.
    """
    return set(np.unique(graph))
