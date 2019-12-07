import numpy as np
from causaldag import DAG, GaussDAG
import itertools as itr


def _coin(p, size=1):
    return np.random.binomial(1, p, size=size)


def unif_away_zero(low=.25, high=1, size=1, all_positive=False):
    if all_positive:
        return np.random.uniform(low, high, size=size)
    return (_coin(.5, size) - .5)*2 * np.random.uniform(low, high, size=size)


def directed_erdos(nnodes, density, size=1):
    """
    Generate random Erdos-Renyi DAG(s) on `nnodes` nodes with density `density`.

    Parameters
    ----------
    nnodes:
        Number of nodes in each graph.
    density:
        Probability of any edge.
    size:
        Number of graphs.

    Examples
    --------
    >>> d = cd.rand.directed_erdos(5, .5)
    """
    if size == 1:
        bools = _coin(density, size=int(nnodes*(nnodes-1)/2))
        arcs = {(i, j) for (i, j), b in zip(itr.combinations(range(nnodes), 2), bools) if b}
        return DAG(nodes=set(range(nnodes)), arcs=arcs)
    else:
        return [directed_erdos(nnodes, density) for _ in range(size)]


def rand_weights(dag, rand_weight_fn=unif_away_zero):
    """
    Generate a GaussDAG from a DAG, with random edge weights independently drawn from `rand_weight_fn`.

    Parameters
    ----------
    dag:
        DAG
    rand_weight_fn:
        Function to generate random weights.

    Examples
    --------
    >>> d = cd.DAG(arcs={(1, 2), (2, 3)})
    >>> g = cd.rand.rand_weights(d)
    """
    weights = rand_weight_fn(size=len(dag.arcs))
    return GaussDAG(nodes=list(range(len(dag.nodes))), arcs=dict(zip(dag.arcs, weights)))


__all__ = ['directed_erdos', 'rand_weights', 'unif_away_zero']



