from typing import Dict, FrozenSet, NewType, Union
import numpy as np
from ...classes.dag import DAG
import itertools as itr
from pyrsistent import pvector


def perm2dag(suffstat: np.ndarray, perm: list, ci_test, ind_test, alpha=.05):
    """
    Given observational data and a node ordering, returns a minimal I-map for the distribution.
    The arrow perm[i] -> perm[j] is present iff i < j and j and i are not dependent given all other predecessors of j.

    :param suffstat:
    :param perm: Permutation used to construct minimal I-map.
    :param ci_test: Conditional independence test.
    :param ind_test: Unconditional independence test.
    :param alpha: Significance level for independence tests.
    :return:
    """
    d = DAG(nodes=set(range(len(perm))))
    for (i, pi_i), (j, pi_j) in itr.combinations(enumerate(perm), 2):
        rest = perm[:j]
        del rest[i]
        if len(rest) == 0:
            p = ind_test(pi_i, pi_j, suffstat)['p_value']
        else:
            p = ci_test(pi_i, pi_j, rest, suffstat)['p_value']
        if p < alpha:
            d.add_arc(pi_i, pi_j)
    return d


def _reverse_arc(dag, covered_arcs, i, j, samples, ci_test, ind_test, alpha):
    # only change what comes before i and j so only effect set of arcs going into i and j
    # anything that's not a parent can't become a parent
    new_dag = dag.copy()
    new_covered_arcs = covered_arcs.copy()
    parents = dag.parents_of(i)

    new_dag.reverse_arc(i, j)
    if parents:
        for parent in parents:
            rest = parents - {parent}
            p_i = ci_test(i, parent, [*rest, j])[2]
            if p_i > alpha:
                new_dag.remove_arc(parent, i)

            if len(rest) == 0:
                p_j = ind_test(samples[:, j], samples[:, parent])[2]
            else:
                p_j = ci_test(samples[:, i], samples[:, parent], samples[:, list(rest)])[2]
            if p_j > alpha:
                new_dag.remove_arc(parent, j)

    for k, l in covered_arcs:
        if k == i or l == i or k == j or l == j:
            if not new_dag.has_arc(k, l):
                new_covered_arcs.remove((k, l))
            elif new_dag.parents_of(k) != new_dag.parents_of(l) - {k}:
                new_covered_arcs.remove((k, l))

    return new_dag, new_covered_arcs


def gsp(suffstat: np.ndarray, starting_perm: list, ci_test, ind_test, alpha: float=0.05, depth: int=4, verbose=False):
    """
    Use the Greedy Sparsest Permutation (GSP) algorithm to estimate the Markov equivalence class of the data-generating
    DAG.

    :param samples:
    :param starting_perm:
    :param ci_test: Conditional independence test.
    :param ind_test: Unconditional independence test.
    :param alpha: Significance level for independence tests.
    :param depth: Maximum depth in depth-first search.
    :param verbose:
    :return:
    """
    # === STARTING VALUES
    current_dag = perm2dag(suffstat, starting_perm, ci_test, ind_test, alpha=alpha)
    if verbose: print("=== STARTING DAG:", current_dag)
    current_covered_arcs = current_dag.reversible_arcs()
    next_dags = [
        _reverse_arc(current_dag, current_covered_arcs, i, j, suffstat, ci_test, ind_test, alpha=alpha)
        for i, j in current_covered_arcs
    ]

    # === RECORDS FOR DEPTH-FIRST SEARCH
    all_visited_dags = set()
    trace = []

    # === SEARCH!
    while True:
        all_visited_dags.add(frozenset(current_dag.arcs))
        lower_dags = [(d, cov_arcs) for d, cov_arcs in next_dags if len(d.arcs) < len(current_dag.arcs)]

        if (len(next_dags) > 0 and len(trace) != depth) or len(lower_dags) > 0:
            if len(lower_dags) > 0:  # start over at lower DAG
                all_visited_dags = set()
                trace = []
                current_dag, current_covered_arcs = lower_dags.pop()
                if verbose: print("=== FOUND DAG WITH FEWER ARCS:", current_dag)
            else:
                trace.append((current_dag, current_covered_arcs, next_dags))
                current_dag, current_covered_arcs = next_dags.pop()
            next_dags = [
                _reverse_arc(current_dag, current_covered_arcs, i, j, suffstat, ci_test, ind_test, alpha=alpha)
                for i, j in current_covered_arcs
            ]
            next_dags = [(d, cov_arcs) for d, cov_arcs in next_dags if frozenset(d.arcs) not in all_visited_dags]
        else:
            if len(trace) == 0:  # reached minimum within search depth
                break
            else:  # backtrack
                current_dag, current_covered_arcs, next_dags = trace.pop()

    return current_dag


def igsp(samples: Dict[FrozenSet, np.ndarray], starting_perm: list):
    pass


def is_icovered(samples: Dict[FrozenSet, np.ndarray], i: int, j: int, ci_test, alpha: float=0.05):
    """
    Tell if an edge i->j is I-covered.

    True if, for all I s.t. i \in I, the interventional distribution of j given i is equal to the observational
    distribution of j given i.

    :param samples:
    :param i: Source of arrow.
    :param j: Target of arrow.
    :param ci_test: Conditional independence test
    :param alpha: Significance level used for conditional independence test
    :return:
    """
    obs_samples = samples[frozenset()]
    num_obs_samples = obs_samples.shape[0]

    is_icov = True
    for iv_nodes, iv_samples in samples.items():
        # print(iv_nodes, iv_samples)
        if i in iv_nodes:
            num_iv_samples = iv_samples.shape[0]
            i_vec = np.concatenate((iv_samples[:, i], obs_samples[:, i]))
            j_vec = np.concatenate((iv_samples[:, j], obs_samples[:, j]))
            labels = np.concatenate((np.zeros(num_obs_samples), np.ones(num_iv_samples)))

            _, _, p = ci_test(i_vec, j_vec, labels)

            if p < alpha:
                is_icov = False

    return is_icov


def unknown_target_igsp(
        samples: Dict[FrozenSet, np.ndarray],
        starting_perm: list,
        ci_test,
        ind_test,
        alpha: float=0.05,
        depth: int=4,
        verbose: bool=False
):
    # === STARTING VALUES
    current_dag = perm2dag(samples[frozenset()], starting_perm, ci_test, ind_test, alpha=alpha)
    if verbose: print("=== STARTING DAG:", current_dag)
    current_covered_arcs = current_dag.reversible_arcs()
    current_i_covered_arcs = [(i, j) for i, j in current_covered_arcs if is_icovered(samples, i, j, ci_test)]
    if verbose: print("=== STARTING I-COVERED ARCS:", current_i_covered_arcs)

    # === RECORDS FOR DEPTH-FIRST SEARCH
    all_visited_dags = set()
    trace = []

    # === SEARCH!
    while True:
        all_visited_dags.add(frozenset(current_dag.arcs))

        # HERE'S THE TRICKY PART - THIS WON'T WORK OUTSIDE THE MARKOV EQUIVALENCE CLASS
        next_dags = [current_dag.copy().reverse_arc(i, j, ignore_error=True) for i, j in current_i_covered_arcs]
        lower_dags = None

        if len(lower_dags) > 0:  # restart at lowest DAG
            all_visited_dags = set()
            trace = []
            current_dag, current_i_covered_arcs = lower_dags[0]
            if verbose: print("FOUND DAG WITH LOWER SCORE")
        elif len(next_dags) > 0 and len(trace) != depth:
            trace.append((current_dag, current_i_covered_arcs))
            current_dag = next_dags[0]
        # === DEAD END ===
        elif len(trace) == 0:  # reached minimum within search depth
            break
        else:  # backtrack
            current_dag, current_i_covered_arcs = trace.pop()

    return current_dag









