import numpy as np
from collections import defaultdict
import random
from causaldag import DAG, GaussDAG, SampleDAG
import itertools as itr
from typing import Union, List, Callable, Optional, Any
from networkx import barabasi_albert_graph, fast_gnp_random_graph
from scipy.special import comb
from tqdm import tqdm
from functools import partial

# class RandWeightFn(Protocol):
#     def __call__(self, size: int) -> Union[float, List[float]]: ...


RandWeightFn = Any


def _coin(p, size=1):
    return np.random.binomial(1, p, size=size)


def unif_away_zero(low=.25, high=1, size=1, all_positive=False):
    if all_positive:
        return np.random.uniform(low, high, size=size)
    signs = (_coin(.5, size) - .5) * 2
    return signs * np.random.uniform(low, high, size=size)


def unif_away_original(original, dist_original=.25, low=.25, high=1):
    if dist_original < low:
        raise ValueError(
            "the lowest absolute value of weights must be larger than the distance between old weights and new weights")
    regions = []
    if original < 0:
        regions.append((low, high))
        if original - dist_original >= -high:
            regions.append((-high, original - dist_original))
        if original + dist_original <= -low:
            regions.append((original + dist_original, -low))
    else:
        regions.append((-high, -low))
        if original + dist_original <= high:
            regions.append((original + dist_original, high))
        if original - dist_original >= low:
            regions.append((low, original - dist_original))
    a, b = random.choices(regions, weights=[b - a for a, b in regions])[0]
    return np.random.uniform(a, b)


def directed_erdos(nnodes, density=None, exp_nbrs=None, size=1, as_list=False, random_order=True) -> Union[
    DAG, List[DAG]]:
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
    >>> import causaldag as cd
    >>> d = cd.rand.directed_erdos(5, .5)
    """
    assert density is not None or exp_nbrs is not None
    density = density if density is not None else exp_nbrs / (nnodes - 1)
    if size == 1:
        # if density < .01:
        #     print('here')
        #     random_nx = fast_gnp_random_graph(nnodes, density, directed=True)
        #     d = DAG(nodes=set(range(nnodes)), arcs=random_nx.edges)
        #     return [d] if as_list else d
        bools = _coin(density, size=int(nnodes * (nnodes - 1) / 2))
        arcs = {(i, j) for (i, j), b in zip(itr.combinations(range(nnodes), 2), bools) if b}
        d = DAG(nodes=set(range(nnodes)), arcs=arcs)
        if random_order:
            nodes = list(range(nnodes))
            d = d.rename_nodes(dict(enumerate(np.random.permutation(nodes))))
        return [d] if as_list else d
    else:
        return [directed_erdos(nnodes, density, random_order=random_order) for _ in range(size)]


def directed_erdos_with_confounders(
        nnodes: int,
        density: Optional[float] = None,
        exp_nbrs: Optional[float] = None,
        num_confounders: int = 1,
        confounder_pervasiveness: float = 1,
        size=1,
        as_list=False,
        random_order=True
) -> Union[DAG, List[DAG]]:
    assert density is not None or exp_nbrs is not None
    density = density if density is not None else exp_nbrs / (nnodes - 1)

    if size == 1:
        confounders = list(range(num_confounders))
        nonconfounders = list(range(num_confounders, nnodes+num_confounders))
        bools = _coin(confounder_pervasiveness, size=int(num_confounders*nnodes))
        confounder_arcs = {(i, j) for (i, j), b in zip(itr.product(confounders, nonconfounders), bools) if b}
        bools = _coin(density, size=int(nnodes * (nnodes - 1) / 2))
        local_arcs = {(i, j) for (i, j), b in zip(itr.combinations(nonconfounders, 2), bools) if b}
        d = DAG(nodes=set(range(nnodes)), arcs=confounder_arcs|local_arcs)

        if random_order:
            nodes = list(range(nnodes+num_confounders))
            d = d.rename_nodes(dict(enumerate(np.random.permutation(nodes))))

        return [d] if as_list else d
    else:
        return [
            directed_erdos_with_confounders(
                nnodes,
                density,
                num_confounders=num_confounders,
                confounder_pervasiveness=confounder_pervasiveness,
                random_order=random_order
            )
            for _ in range(size)
        ]


def rand_weights(dag, rand_weight_fn: RandWeightFn = unif_away_zero) -> GaussDAG:
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
    >>> import causaldag as cd
    >>> d = cd.DAG(arcs={(1, 2), (2, 3)})
    >>> g = cd.rand.rand_weights(d)
    """
    weights = rand_weight_fn(size=len(dag.arcs))
    return GaussDAG(nodes=list(range(len(dag.nodes))), arcs=dict(zip(dag.arcs, weights)))


def _leaky_relu(vals):
    return np.where(vals > 0, vals, vals * .01)


def rand_nn_functions(
        dag: DAG,
        num_layers=3,
        nonlinearity=_leaky_relu,
        noise=lambda: np.random.laplace(0, 1)
) -> SampleDAG:
    s = SampleDAG(dag._nodes, arcs=dag._arcs)

    # for each node, create the conditional
    for node in dag._nodes:
        nparents = dag.indegree(node)
        layer_mats = [np.random.rand(nparents, nparents) * 2 for _ in range(num_layers)]

        def conditional(parent_vals):
            vals = parent_vals
            for a in layer_mats:
                vals = a @ vals
                vals = nonlinearity(vals)
            return vals + noise()

        s.set_conditional(node, conditional)

    return s


def _cam_conditional(parent_vals, c_node, parent_weights, parent_bases, noise):
    return sum([
        c_node * weight * base(val) for weight, base, val in zip(parent_weights, parent_bases, parent_vals)
    ]) + noise()


def rand_additive_basis(
        dag: DAG,
        basis: list,
        snr_dict: Optional[dict] = None,
        rand_weight_fn: RandWeightFn = unif_away_zero,
        noise=lambda: np.random.normal(0, 1),
        internal_variance: int = 1,
        num_monte_carlo: int = 10000,
        progress=False
):
    """
    Generate a random structural causal model (SCM), using `dag` as the structure, and with each variable
    being a general additive model (GAM) of its parents.

    Parameters
    ----------
    dag:
        A DAG to use as the structure for the model.
    basis:
        Basis functions for the GAM.
    snr_dict:
        A dictionary mapping each number of parents to the desired signal-to-noise ratio (SNR) for nodes
        with that many parents. By default, 1/2 for any number of parents.
    rand_weight_fn:
        A function to generate random weights for each parent.
    noise:
        A function to generate random internal noise for each node.
    internal_variance:
        The variance of the above noise function.
    num_monte_carlo:
        The number of Monte Carlo samples used when computing coefficients to achieve the desired SNR.

    Examples
    --------
    >>> import causaldag as cd
    >>> import numpy as np
    >>> d = cd.DAG(arcs={(1, 2), (2, 3), (1, 3)})
    >>> basis = [np.sin, np.cos, np.exp]
    >>> snr_dict = {1: 1/2, 2: 2/3}
    >>> g = cd.rand.rand_additive_basis(d, basis, snr_dict)
    """
    if snr_dict is None:
        snr_dict = {nparents: 1/2 for nparents in range(dag.nnodes)}

    sample_dag = SampleDAG(dag._nodes, arcs=dag._arcs)
    top_order = dag.topological_sort()
    sample_dict = defaultdict(list)

    # for each node, create the conditional
    node_iterator = top_order if not progress else tqdm(top_order)
    for node in node_iterator:
        parents = dag.parents_of(node)
        nparents = dag.indegree(node)
        parent_bases = random.choices(basis, k=nparents)
        parent_weights = rand_weight_fn(size=nparents)

        c_node = None
        if nparents > 0:
            values_from_parents = []
            for i in range(num_monte_carlo):
                val = sum([
                    weight * base(sample_dict[parent][i])
                    for weight, base, parent in zip(parent_weights, parent_bases, parents)
                ])
                values_from_parents.append(val)
            variance_from_parents = np.var(values_from_parents)

            try:
                desired_snr = snr_dict[nparents]
            except ValueError:
                raise Exception(f"`snr_dict` does not specify a desired SNR for nodes with {nparents} parents")
            c_node = internal_variance / variance_from_parents * desired_snr / (1 - desired_snr)

        conditional = partial(_cam_conditional, c_node=c_node, parent_weights=parent_weights, parent_bases=parent_bases, noise=noise)

        for i in range(num_monte_carlo):
            val = conditional([sample_dict[parent][i] for parent in parents])
            sample_dict[node].append(val)
        sample_dag.set_conditional(node, conditional)

    return sample_dag


# OPTION 1
# - equally predictable given parents (signal to noise ratio): internal variance \propto variance from the parents
# - compute variance of each parent. add up. set my internal noise variance \propto variance from parents
# - always keep noise variance same, scale signal coefficients

# OPTION 2
# - bound each variable

# OPTION 3


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


def alter_weights(
        gdag: GaussDAG,
        prob_altered: float = None,
        num_altered: int = None,
        prob_added: float = None,
        num_added: int = None,
        prob_removed: float = None,
        num_removed: int = None,
        rand_weight_fn=unif_away_zero,
        rand_change_fn=unif_away_original
):
    """
    Return a copy of a GaussDAG with some of its arc weights randomly altered by `rand_weight_fn`.

    Parameters
    ----------
    gdag:
        GaussDAG
    prob_altered:
        Probability each arc has its weight altered.
    num_altered:
        Number of arcs whose weights are altered.
    prob_added:
        Probability that each missing arc is added.
    num_added:
        Number of missing arcs added.
    prob_removed:
        Probability that each arc is removed.
    num_removed:
        Number of arcs removed.
    rand_weight_fn:
        Function that returns a random weight for each new edge.
    rand_change_fn:
        Function that takes the current weight of an edge and returns the new weight.
    """
    if num_altered is None and prob_altered is None:
        raise ValueError("Must specify at least one of `prob_altered` or `num_altered`.")
    if num_added is None and prob_added is None:
        raise ValueError("Must specify at least one of `prob_added` or `num_added`.")
    if num_removed is None and prob_removed is None:
        raise ValueError("Must specify at least one of `prob_removed` or `num_removed`.")
    if num_altered + num_removed > gdag.num_arcs:
        raise ValueError(
            f"Tried altering {num_altered} arcs and removing {num_removed} arcs, but there are only {gdag.num_arcs} arcs in this DAG.")
    num_missing_arcs = comb(gdag.nnodes, 2) - gdag.num_arcs
    if num_added > num_missing_arcs:
        raise ValueError(
            f"Tried adding {num_added} arcs but there are only {num_missing_arcs} arcs missing from the DAG.")

    # GET NUMBER ADDED/CHANGED/REMOVED
    num_altered = num_altered if num_altered is not None else np.random.binomial(gdag.num_arcs, prob_altered)
    num_removed = num_removed if num_removed is not None else np.random.binomial(gdag.num_arcs, prob_removed)
    num_removed = min(num_removed, gdag.num_arcs - num_altered)
    num_added = num_added if num_added is not None else np.random.binomial(num_missing_arcs, prob_added)

    # GET ACTUAL ARCS THAT ARE ADDED/CHANGED/REMOVED
    altered_arcs = random.sample(list(gdag.arcs), num_altered)
    removed_arcs = random.sample(list(gdag.arcs - set(altered_arcs)), num_removed)
    valid_arcs_to_add = set(itr.combinations(gdag.topological_sort(), 2)) - gdag.arcs
    added_arcs = random.sample(list(valid_arcs_to_add), num_added)

    # CREATE NEW DAG
    new_gdag = gdag.copy()
    weights = gdag.arc_weights
    for i, j in altered_arcs:
        new_gdag.set_arc_weight(i, j, rand_change_fn(weights[(i, j)]))
    for i, j in removed_arcs:
        new_gdag.remove_arc(i, j)
    new_weights = rand_weight_fn(size=num_added)
    for (i, j), val in zip(added_arcs, new_weights):
        new_gdag.set_arc_weight(i, j, val)

    return new_gdag


__all__ = [
    'directed_erdos',
    'directed_erdos_with_confounders',
    'rand_weights',
    'unif_away_zero',
    'directed_barabasi',
    'directed_random_graph',
    'rand_nn_functions',
    'unif_away_original',
    'alter_weights',
    'rand_additive_basis'
]
