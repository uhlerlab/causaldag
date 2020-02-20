import numpy as np
from causaldag import DAG, GaussDAG, SampleDAG
import itertools as itr
from typing import Union, List, Callable
from networkx import barabasi_albert_graph


def _coin(p, size=1):
    return np.random.binomial(1, p, size=size)


def unif_away_zero(low=.25, high=1, size=1, all_positive=False):
    if all_positive:
        return np.random.uniform(low, high, size=size)
    return (_coin(.5, size) - .5) * 2 * np.random.uniform(low, high, size=size)


def directed_erdos(nnodes, density, size=1, as_list=False) -> Union[DAG, List[DAG]]:
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
    as_list:
        If True, always return as a list, even if only one DAG is generated.

    Examples
    --------
    >>> d = cd.rand.directed_erdos(5, .5)
    """
    if size == 1:
        bools = _coin(density, size=int(nnodes * (nnodes - 1) / 2))
        arcs = {(i, j) for (i, j), b in zip(itr.combinations(range(nnodes), 2), bools) if b}
        d = DAG(nodes=set(range(nnodes)), arcs=arcs)
        return [d] if as_list else d
    else:
        return [directed_erdos(nnodes, density) for _ in range(size)]


def rand_weights(dag, rand_weight_fn=unif_away_zero) -> GaussDAG:
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


def rand_nn_functions(dag: DAG, num_layers=3) -> SampleDAG:
    s = SampleDAG(dag._nodes, arcs=dag._arcs)
    for node in dag._nodes:
        def conditional(parent_vals):
            p = len(parent_vals)
            vals = parent_vals
            for _ in range(num_layers):
                a = np.random.random((p, p))*2
                vals = a @ vals
                vals = np.where(vals > 0, vals, vals*.01)
            return np.random.random(p)*2 @ vals + np.random.laplace(0, 1)
        s.set_conditional(node, conditional)
    return s


def directed_random_graph(nnodes: int, random_graph_model: Callable, size=1, as_list=False) -> Union[DAG, List[DAG]]:
    if size == 1:
        # generate a random undirected graph
        edges = random_graph_model(nnodes).edges

        # generate a random permutation
        random_permutation = np.arange(nnodes)
        np.random.shuffle(random_permutation)

        arcs = []
        for edge in edges:
            node1, node2 = edge
            node1_position = np.where(random_permutation == node1)[0][0]
            node2_position = np.where(random_permutation == node2)[0][0]
            if node1_position < node2_position:
                source = node1
                endpoint = node2
            else:
                source = node2
                endpoint = node1
            arcs.append((source, endpoint))
        d = DAG(nodes=set(range(nnodes)), arcs=arcs)
        return [d] if as_list else d
    else:
        return [directed_random_graph(nnodes, random_graph_model) for _ in range(size)]


def directed_barabasi(nnodes: int, nattach: int, size=1, as_list=False) -> Union[DAG, List[DAG]]:
    random_graph_model = lambda nnodes: barabasi_albert_graph(nnodes, nattach)
    return directed_random_graph(nnodes, random_graph_model, size=size, as_list=as_list)


__all__ = [
    'directed_erdos',
    'rand_weights',
    'unif_away_zero',
    'directed_barabasi',
    'directed_random_graph',
    'rand_nn_functions'
]


