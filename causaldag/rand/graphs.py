import numpy as np
from causaldag import DAG, GaussDAG
import itertools as itr


def _coin(p, size=1):
    return np.random.binomial(1, p, size=size)


def unif_away_zero(low=.25, high=1, size=1):
    return (_coin(.5, size) - .5)*2 * np.random.uniform(low, high)


def directed_erdos(n, s, size=1):
    """Generate random Erdos-Renyi DAGs
    """
    if size == 1:
        bools = _coin(s, size=int(n*(n-1)/2))
        arcs = {(i, j) for (i, j), b in zip(itr.combinations(range(n), 2), bools) if b}
        return DAG(nodes=set(range(n)), arcs=arcs)
    else:
        return [directed_erdos(n, s) for _ in range(size)]


def rand_weights(dag, rand_weight_fn=unif_away_zero):
    """Generate random arc weights for a DAG
    """
    weights = rand_weight_fn(size=len(dag.arcs))
    return GaussDAG(nodes=list(range(len(dag.nodes))), arcs=dict(zip(dag.arcs, weights)))


__all__ = ['directed_erdos', 'rand_weights', 'unif_away_zero']



