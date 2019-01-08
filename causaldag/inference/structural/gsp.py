from typing import Dict, FrozenSet, Optional, Any
import numpy as np
from ...classes.dag import DAG
import itertools as itr
from ...utils.ci_tests import CI_Test, InvarianceTest
import random


def perm2dag(suffstat: Any, perm: list, ci_test: CI_Test, alpha: float=.05):
    """
    Given observational data and a node ordering, returns a minimal I-map for the distribution.
    The arrow perm[i] -> perm[j] is present iff i < j and j and i are not dependent given all other predecessors of j.

    :param suffstat:
    :param perm: Permutation used to construct minimal I-map.
    :param ci_test: Conditional independence test.
    :param alpha: Significance level for independence tests.
    :return:
    """
    d = DAG(nodes=set(range(len(perm))))
    for (i, pi_i), (j, pi_j) in itr.combinations(enumerate(perm), 2):
        rest = perm[:j]
        del rest[i]
        test_results = ci_test(suffstat, pi_i, pi_j, cond_set=rest if len(rest) != 0 else None, alpha=alpha)
        if test_results['reject']:  # add arc if we reject the hypothesis that the nodes are conditionally independent
            d.add_arc(pi_i, pi_j)
    return d


def _reverse_arc_gsp(dag, covered_arcs, i, j, suffstat, ci_test, alpha):
    """
    Given a minimal I-map and its covered arcs, reverse the arc i->j and find the new minimal I-map using conditional
    independence test.
    """
    # only change what comes before i and j so only effect set of arcs going into i and j
    # anything that's not a parent can't become a parent
    new_dag = dag.copy()
    parents = dag.parents_of(i)

    new_dag.reverse_arc(i, j)
    if parents:
        for parent in parents:
            rest = parents - {parent}
            p_i = ci_test(suffstat, i, parent, [*rest, j], alpha=alpha)['p_value']
            if p_i > alpha:
                new_dag.remove_arc(parent, i)

            p_j = ci_test(suffstat, j, parent, cond_set=[*rest] if len(rest) != 0 else None, alpha=alpha)['p_value']
            if p_j > alpha:
                new_dag.remove_arc(parent, j)

    # TODO: I DON'T THINK THIS IS MUCH FASTER THAN JUST RECOMPUTING THE COVERED ARCS
    new_covered_arcs = covered_arcs.copy() - dag.incident_arcs(i) - dag.incident_arcs(j)
    for k, l in new_dag.incident_arcs(i) | new_dag.incident_arcs(j):
        if new_dag.parents_of(k) == new_dag.parents_of(l) - {k}:
            new_covered_arcs.add((k, l))

    return new_dag, new_covered_arcs


def gsp(
        suffstat: Dict,
        nnodes: int,
        ci_test: CI_Test,
        alpha: float=0.01,
        depth: Optional[int]=4,
        nruns: int=5,
        verbose=False
) -> DAG:
    """
    Use the Greedy Sparsest Permutation (GSP) algorithm to estimate the Markov equivalence class of the data-generating
    DAG.

    :param suffstat:
    :param nnodes:
    :param ci_test: Conditional independence test.
    :param alpha: Significance level for independence tests.
    :param depth: Maximum depth in depth-first search. Use None for infinite search depth.
    :param nruns: Number of runs of the algorithm. Each run starts at a random permutation and the sparsest DAG from all
    runs is returned.
    :param verbose:
    :return:
    """
    min_dag = None
    for r in range(nruns):
        # === STARTING VALUES
        starting_perm = random.sample(list(range(nnodes)), nnodes)
        current_dag = perm2dag(suffstat, starting_perm, ci_test, alpha=alpha)
        if verbose: print("=== STARTING DAG:", current_dag)
        current_covered_arcs = current_dag.reversible_arcs()
        next_dags = [
            _reverse_arc_gsp(current_dag, current_covered_arcs, i, j, suffstat, ci_test, alpha=alpha)
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

                # covered_arcs_check = current_dag.reversible_arcs()
                # if current_covered_arcs != covered_arcs_check:
                #     print(current_covered_arcs, covered_arcs_check)
                #     raise Exception

                next_dags = [
                    _reverse_arc_gsp(current_dag, current_covered_arcs, i, j, suffstat, ci_test, alpha=alpha)
                    for i, j in current_covered_arcs
                ]
                next_dags = [(d, cov_arcs) for d, cov_arcs in next_dags if frozenset(d.arcs) not in all_visited_dags]
            else:
                if len(trace) == 0:  # reached minimum within search depth
                    break
                else:  # backtrack
                    current_dag, current_covered_arcs, next_dags = trace.pop()

        if min_dag is None or len(current_dag.arcs) < len(min_dag.arcs):
            min_dag = current_dag

    return min_dag


def igsp(samples: Dict[FrozenSet, np.ndarray], starting_perm: list):
    pass


def is_icovered(
        samples: Dict[FrozenSet, np.ndarray],
        i: int,
        j: int,
        dag: DAG,
        invariance_test: InvarianceTest,
        alpha: float=0.05
):
    """
    Tell if an edge i->j is I-covered.

    True if, for all I s.t. i \in I, the distribution of j given its parents varies between the observational and
    interventional data.

    :param samples:
    :param i: Source of arrow.
    :param j: Target of arrow.
    :param dag: DAG
    :param invariance_test: Invariance test for conditional distributions.
    :param alpha: Significance level used for invariance test. Note: this does NOT control the false negative rate for
    this test.
    :return:
    """
    obs_samples = samples[frozenset()]
    parents_j = list(dag.parents_of(j))

    is_icov = True
    for iv_nodes, iv_samples in samples.items():
        if i in iv_nodes:
            test_results = invariance_test(obs_samples, iv_samples, j, cond_set=parents_j, alpha=alpha)

            # can't reject the null hypothesis that the conditional distribution is invariant
            if test_results['p_value'] > alpha:
                is_icov = False

    return is_icov


def unknown_target_igsp(
        samples: Dict[FrozenSet, np.ndarray],
        suffstat: Any,
        nnodes: int,
        ci_test: CI_Test,
        invariance_test: InvarianceTest,
        alpha: float=0.01,
        alpha_invariance: float=0.05,
        depth: Optional[int]=4,
        nruns: int=5,
        verbose: bool=False
) -> DAG:
    """
    Use the Unknown Target Greedy Sparsest Permutation algorithm to estimate a DAG in the I-MEC of the data-generating
    DAG.

    :param samples:
    :param suffstat:
    :param nnodes:
    :param ci_test: Conditional independence test
    :param invariance_test: Conditional independence test used for determining
    invariance of conditional distributions to interventions.
    :param alpha: Significance level for conditional independence test
    :param alpha_invariance: Significance level for invariance test
    :param depth: Maximum depth of search. Use None for infinite search depth.
    :param nruns: Number of runs of the algorithm. Each run starts at a random permutation and the lowest-scoring
    DAG from all runs is returned.
    :param verbose:
    :return:
    """
    # === DICTIONARY CACHING RESULTS OF ALL INVARIANCE TESTS SO FAR
    is_variant_dict = {iv_nodes: dict() for iv_nodes in samples if iv_nodes != frozenset}
    obs_samples = samples[frozenset()]

    # === HELPER METHODS
    def _get_is_variant(iv_nodes, j, cond_set):
        """
        Check if in the intervention on iv_nodes, the conditional distribution of j given cond_set is the same as
        the observational distribution. Cache through is_variant_dict.
        """
        is_variant = is_variant_dict[iv_nodes].get((j, cond_set))
        if is_variant is None:
            is_variant = invariance_test(
                obs_samples, samples[iv_nodes], j, cond_set=list(cond_set), alpha=alpha_invariance)['reject']
            is_variant_dict[iv_nodes][(j, cond_set)] = is_variant
        return is_variant

    def _is_icovered(i, j, dag):
        """
        Check if the edge i->j is I-covered in the DAG dag
        """
        parents_j = frozenset(dag.parents_of(j))
        is_icov = True
        for iv_nodes, iv_samples in samples.items():
            if i in iv_nodes:
                if not _get_is_variant(iv_nodes, j, parents_j):
                    is_icov = False
        return is_icov

    def _get_num_variant(dag):
        """
        Count the number of variances for the DAG dag
        """
        num_variant = 0
        for i in dag.nodes:
            parents_i = frozenset(dag.parents_of(i))
            for iv_nodes, iv_samples in samples.items():
                if iv_nodes != frozenset():
                    if _get_is_variant(iv_nodes, i, parents_i): num_variant += 1

        return num_variant

    def _reverse_arc_igsp(dag, i_covered_arcs, i, j):
        """
        Return the DAG that comes from reversing the arc i->j, as well as its I-covered arcs and its score
        """
        new_dag = dag.copy()
        parents = dag.parents_of(i)

        new_dag.reverse_arc(i, j)
        if parents:
            for parent in parents:
                rest = parents - {parent}
                p_i = ci_test(suffstat, i, parent, [*rest, j], alpha=alpha)['p_value']
                if p_i > alpha:
                    new_dag.remove_arc(parent, i)

                p_j = ci_test(suffstat, j, parent, cond_set=[*rest] if len(rest) != 0 else None, alpha=alpha)['p_value']
                if p_j > alpha:
                    new_dag.remove_arc(parent, j)

        # new_i_covered_arcs = i_covered_arcs.copy() - dag.incident_arcs(i) - dag.incident_arcs(j)
        # for k, l in new_dag.incident_arcs(i) | new_dag.incident_arcs(j):
        #     if new_dag.parents_of(k) == new_dag.parents_of(l) - {k} and _is_icovered(i, j, dag):
        #         new_i_covered_arcs.add((k, l))

        new_covered_arcs = new_dag.reversible_arcs()
        new_i_covered_arcs = [(i, j) for i, j in new_covered_arcs if _is_icovered(i, j, current_dag)]
        new_score = len(new_dag.arcs) + _get_num_variant(dag)

        return new_dag, new_i_covered_arcs, new_score

    # === MINIMUM DAG AND SCORE FOUND BY ANY RUN
    min_dag = None
    min_score = float('inf')

    # === MULTIPLE RUNS
    for r in range(nruns):
        # === STARTING VALUES
        starting_perm = random.sample(list(range(nnodes)), nnodes)
        current_dag = perm2dag(suffstat, starting_perm, ci_test, alpha=alpha)
        current_score = len(current_dag.arcs) + _get_num_variant(current_dag)
        if verbose: print("=== STARTING DAG:", current_dag, "== SCORE:", current_score)

        current_covered_arcs = current_dag.reversible_arcs()
        current_i_covered_arcs = [(i, j) for i, j in current_covered_arcs if _is_icovered(i, j, current_dag)]
        if verbose: print("=== STARTING I-COVERED ARCS:", current_i_covered_arcs)
        next_dags = [_reverse_arc_igsp(current_dag, current_i_covered_arcs, i, j) for i, j in current_i_covered_arcs]
        next_dags = [(d, i_cov_arcs, score) for d, i_cov_arcs, score in next_dags if score <= current_score]

        # === RECORDS FOR DEPTH-FIRST SEARCH
        all_visited_dags = set()
        trace = []

        # === SEARCH!
        while True:
            # print("current score: ", current_score)
            # print("next dags:", [str(d[0]) for d in next_dags])
            all_visited_dags.add(frozenset(current_dag.arcs))
            lower_dags = [(d, i_cov_arcs, score) for d, i_cov_arcs, score in next_dags if score < current_score]

            if (len(next_dags) > 0 and len(trace) != depth) or len(lower_dags) > 0:
                if len(lower_dags) > 0:  # restart at a lower DAG
                    all_visited_dags = set()
                    trace = []
                    current_dag, current_i_covered_arcs, current_score = lower_dags.pop()
                    if verbose: print("FOUND DAG WITH LOWER SCORE:", current_dag, "== SCORE:", current_score)
                else:
                    trace.append((current_dag, current_i_covered_arcs, next_dags))
                    current_dag, current_i_covered_arcs, current_score = next_dags.pop()
                next_dags = [_reverse_arc_igsp(current_dag, current_i_covered_arcs, i, j)for i, j in current_i_covered_arcs]
                next_dags = [(d, i_cov_arcs, score) for d, i_cov_arcs, score in next_dags if score <= current_score]
                next_dags = [(d, i_cov_arcs, score) for d, i_cov_arcs, score in next_dags if frozenset(d.arcs) not in all_visited_dags]
            # === DEAD END ===
            else:
                if len(trace) == 0:  # reached minimum within search depth
                    break
                else:  # backtrack
                    current_dag, current_i_covered_arcs, next_dags = trace.pop()

        if min_dag is None or current_score < min_score:
            min_dag = current_dag
            min_score = current_score
        if verbose: print("=== FINISHED RUN %s/%s ===" % (r+1, nruns))

    return min_dag








