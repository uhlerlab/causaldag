import numpy as np
from sklearn.utils.random import sample_without_replacement
import random
import networkx as nx


def bootstrap_generator(n_bootstrap_iterations, sample_fraction, X, random_state=None):
    """Generates bootstrap samples from dataset."""
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    n_samples = len(X)
    n_subsamples = np.floor(sample_fraction * n_samples).astype(int)
    for _ in range(n_bootstrap_iterations):
        subsample = sample_without_replacement(n_samples, n_subsamples)
        yield subsample


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

def adjacency2edges(adjacency_matrix):
    """Returns a set of edges for a given adjacency matrix."""
    g = nx.from_numpy_matrix(adjacency_matrix)
    edges = {frozenset({i, j}) for i, j in g.edges()}
    return edges