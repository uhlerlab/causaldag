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


def dci_undirected_graph(X1, X2, alpha=1.0, max_iter=1000, edge_threshold=0.05, verbose=0):
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
    """

    if verbose > 0:
        print("Running KLIEP to get difference undirected graph...")
        
    k1 = kernel_linear(X1)
    k2 = kernel_linear(X2)
    theta = naive_subgradient_descent(k1, k2, alpha=alpha, max_iter=1000, verbose=verbose)
    difference_ug = compute_difference_graph(X1, theta, edge_threshold=edge_threshold)
    return difference_ug


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
        iter = iter + 1
        if (verbose > 0) & (iter == max_iter):
            print('Maximum iteration reached')
    return theta


def compute_difference_graph(X, theta, edge_threshold=0.05):
    """
    Obtain difference undirected graph from estimated parameters.
    """
    n, d = X.shape
    delta_ug = np.zeros((d,d))
    delta_ug[np.triu_indices(d, 1)] = theta[0:-d].flatten()
    delta_ug = delta_ug + delta_ug.T
    # remove edges that are below cutoff threshold
    delta_ug[np.abs(delta_ug) < edge_threshold] = 0
    g = nx.from_numpy_matrix(delta_ug)
    return list(g.edges())