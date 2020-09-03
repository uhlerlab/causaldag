from ..rand.graphs import directed_erdos, alter_weights, rand_weights, unif_away_original, unif_away_zero
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
        return_graphs=False,
        rand_weight_fn=unif_away_zero,
        rand_change_fn=unif_away_original
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
    g1 = rand_weights(d1, rand_weight_fn=rand_weight_fn)
    g2 = alter_weights(
        g1,
        num_altered=num_altered,
        num_removed=num_removed,
        num_added=num_added,
        rand_weight_fn=rand_weight_fn,
        rand_change_fn=rand_change_fn
    )
    X1 = g1.sample(nsamples)
    X2 = g2.sample(nsamples)
    difference = set(zip(*np.where(g1.to_amat() != g2.to_amat())))
    difference_ug = set(zip(*np.where(~np.isclose(g1.precision, g2.precision))))
    difference_ug = {frozenset({i, j}) for i, j in difference_ug if i != j}

    if return_graphs:
        return X1, X2, difference, difference_ug, g1, g2
    return X1, X2, difference


