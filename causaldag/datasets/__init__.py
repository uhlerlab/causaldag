from ..rand.graphs import directed_erdos, alter_weights, rand_weights
import numpy as np
import random


def create_synthetic_difference(
        nnodes=10,
        nsamples=1000,
        num_altered=1,
        num_removed=1,
        num_added=1,
        exp_nbrs=2,
        seed=8181818,
        return_graphs=False
):
    """
    Create two synthetic datasets from two related causal graphs.

    nnodes:
        Number of nodes in each graph.
    nsamples:
        Number of samples from each graph.
    num_altered:
        Number of altered
    """
    random.seed(seed)
    np.random.seed(seed)
    d1 = directed_erdos(nnodes, exp_nbrs=min(exp_nbrs, nnodes-1))
    g1 = rand_weights(d1)
    g2 = alter_weights(g1, num_altered=num_altered, num_removed=num_removed, num_added=num_added)
    X1 = g1.sample(nsamples)
    X2 = g2.sample(nsamples)
    difference = set(zip(*np.where(g1.to_amat() != g2.to_amat())))

    if return_graphs:
        return X1, X2, difference, g1, g2
    return X1, X2, difference


