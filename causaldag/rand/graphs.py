import numpy as np
from ..classes.dag import DAG
import itertools as itr
from more_itertools import chunked


def _coin(p, size=1):
    return np.random.binomial(1, p, size=size)


def directed_erdos(n, s, size=1):
    if size == 1:
        bools = _coin(s, size=int(n*(n-1)/2))
        arcs = {(i, j) for (i, j), b in zip(itr.combinations(range(n), 2), bools) if b}
        return DAG(nodes=set(range(n)), arcs=arcs)
    else:
        all_bools = _coin(s, size=int(n*(n-1)/2)*size)
        chunked_bools = chunked(all_bools, size)
        possible_arcs = list(itr.combinations(range(n), 2))
        chunked_arcs = [{(i, j) for (i, j), b in zip(possible_arcs, bools)} for bools in chunked_bools]
        return [DAG(nodes=set(range(n)), arcs=arcs) for arcs in chunked_arcs]


__all__ = ['directed_erdos']



