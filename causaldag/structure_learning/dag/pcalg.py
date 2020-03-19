from causaldag import PDAG
from causaldag import UndirectedGraph
from causaldag.utils.ci_tests import CI_Tester
import itertools as itr


def skeleton(nodes: set, ci_tester: CI_Tester, max_cond_set: int=None, verbose=False):
    """
    Estimate the skeleton of an underlying DAG using the order-independent skeleton estimation method of
    Colombo and Maathius (2014).

    Parameters
    ----------
    nodes:
        Labels of nodes in the graph.
    ci_tester:
        A conditional independence tester, which has a method is_ci taking two sets A and B, and a conditioning set C,
        and returns True/False.
    max_cond_set:
        Maximum size of conditioning set tested to separate nodes.
    verbose:
        If True, print edges as they are removed, along with the separating set responsible for removing them.

    See Also
    --------
    pcalg

    Returns
    -------
    (skeleton, sepset)
    """
    nnodes = len(nodes)
    ug = UndirectedGraph(edges=set(itr.combinations(nodes, 2)))
    sepset = {}
    max_cond_set = max_cond_set if max_cond_set is not None else nnodes-2
    for c_size in range(max_cond_set+1):
        adjacencies = ug.neighbors
        for i, j in itr.permutations(nodes, 2):
            if ug.has_edge(i, j) and len(adjacencies[i] - {j}) >= c_size:
                for cond_set in itr.combinations(adjacencies[i] - {j}, c_size):
                    if ci_tester.is_ci(i, j, cond_set):
                        if verbose: print(f"Removing {i}-{j}, separated by {cond_set}")
                        ug.delete_edge(i, j)
                        sepset[frozenset({i, j})] = cond_set
                        break
    return ug, sepset


def pcalg(
        nodes,
        ci_tester: CI_Tester=None,
        skel=None,
        sepset=None,
        solve_conflict: bool=False,
        max_cond_set: int=None,
        verbose: bool=False
) -> PDAG:
    """
    Use the PC (Peters-Clark) algorithm to estimate the Markov equivalence class of the data-generating DAG.

    Parameters
    ----------
    nodes:
        Labels of nodes in the graph.
    ci_tester:
        A conditional independence tester, which has a method is_ci taking two sets A and B, and a conditioning set C,
        and returns True/False.
    skel:
        An estimated skeleton. If not provided, uses the `skeleton` method to estimate.
    sepset:
        The separating sets for non-adjacent nodes in the estimated skeleton.
    solve_conflict:
        If False, any disagreements on v-structures are simply overwritten. If True, allow both orientations
        (represented by a bidirected edge).
    verbose:
        If True, print decisions made by the algorithm.

    See Also
    --------
    gsp

    Returns
    -------
    est_dag
    """
    if solve_conflict:
        raise NotImplementedError
    if ci_tester is None:
        if skel is None or sepset is None:
            raise ValueError("Must provide either ci_tester or skeleton and sepset dictionary")
    if ci_tester is not None:
        skel, sepset = skeleton(nodes, ci_tester, max_cond_set=max_cond_set, verbose=verbose)
    adjacencies = skel.neighbors

    arcs = set()
    for i, k in itr.combinations(nodes, 2):
        if not skel.has_edge(i, k):
            for j in adjacencies[i] & adjacencies[k]:
                if j not in sepset[frozenset({i, k})]:
                    if not solve_conflict:
                        arcs.discard((j, k))
                        arcs.discard((j, k))
                    arcs.add((i, j))
                    arcs.add((k, j))

    cpdag = PDAG(nodes=nodes, arcs=arcs, edges=skel.edges-{frozenset({*arc}) for arc in arcs})
    cpdag.to_complete_pdag(verbose=verbose, solve_conflict=solve_conflict)

    return cpdag


if __name__ == '__main__':
    import causaldag as cd
    from causaldag.utils.ci_tests import MemoizedCI_Tester, dsep_test

    import numpy as np
    import random
    np.random.seed(9890142)
    random.seed(9890142)

    nnodes = 15
    ngraphs = 10
    exp_nbrs = 3
    dags = cd.rand.directed_erdos(nnodes, exp_nbrs/(nnodes-1), ngraphs)
    nodes = set(range(nnodes))
    ci_testers = [MemoizedCI_Tester(dsep_test, d) for d in dags]
    est_skels = [skeleton(set(range(nnodes)), ci_tester)[0] for ci_tester in ci_testers]
    false_positives = [est_skel.edges - d.skeleton for est_skel, d in zip(est_skels, dags)]
    false_negatives = [d.skeleton - est_skel.edges for est_skel, d in zip(est_skels, dags)]
    print(false_positives)
    print(false_negatives)

    est_cpdags = [pcalg(nodes, ci_tester) for ci_tester in ci_testers]
    true_cpdags = [d.cpdag() for d in dags]
    correct_cpdag = [true_cpdag == est_cpdag for true_cpdag, est_cpdag in zip(true_cpdags, est_cpdags)]
    shds = [true_cpdag.shd(est_cpdag) for true_cpdag, est_cpdag in zip(true_cpdags, est_cpdags)]
    print(correct_cpdag)

