from typing import Dict, Optional, Any, List, Set, Union
from causaldag import DAG
import itertools as itr
from causaldag.utils.ci_tests import CI_Tester, MemoizedCI_Tester
from causaldag.utils.invariance_tests import InvarianceTester
from causaldag.utils.core_utils import powerset
import random
import operator as op
from causaldag.inference.structural.undirected import threshold_ug
from causaldag import UndirectedGraph


def perm2dag(perm, ci_tester: CI_Tester, restricted=True, fixed_adjacencies=set(), fixed_gaps=set()):
    d = DAG(nodes=set(range(len(perm))))
    for (i, pi_i), (j, pi_j) in itr.combinations(enumerate(perm), 2):
        if (pi_i, pi_j) in fixed_adjacencies or (pi_j, pi_i) in fixed_adjacencies:
            d.add_arc(pi_i, pi_j)
            continue
        if (pi_i, pi_j) in fixed_gaps or (pi_j, pi_i) in fixed_gaps:
            continue

        if restricted:
            cond_set = set(perm[i+1:j]) | d.parents_of(pi_i) | d.parents_of(pi_j)
        else:
            cond_set = perm[:j]
            del cond_set[i]

        if not ci_tester.is_ci(pi_i, pi_j, cond_set):
            d.add_arc(pi_i, pi_j)
    return d


def min_degree_alg(undirected_graph, ci_tester: CI_Tester):
    permutation = []
    curr_undirected_graph = undirected_graph
    while curr_undirected_graph._nodes:
        min_degree = min(curr_undirected_graph.degrees.items(), key=op.itemgetter(1))[1]
        min_degree_nodes = {node for node, degree in curr_undirected_graph.degrees.items() if degree == min_degree}
        k = random.sample(min_degree_nodes, 1)[0]
        nbrs_k = curr_undirected_graph._neighbors[k]

        curr_undirected_graph = curr_undirected_graph.copy()
        curr_undirected_graph.delete_node(k)
        for nbr1, nbr2 in itr.combinations(nbrs_k, 2):
            if not curr_undirected_graph.has_edge(nbr1, nbr2):
                curr_undirected_graph.add_edge(nbr1, nbr2)
            elif ci_tester.is_ci(nbr1, nbr2, curr_undirected_graph._nodes - {nbr1, nbr2, k}):
                curr_undirected_graph.delete_edge(nbr1, nbr2)

        permutation.append(k)

    return list(reversed(permutation))


def jci_gsp(
        setting_list: List[Dict],
        nnodes: int,
        combined_ci_tester: CI_Tester,
        nruns: int=5,
        verbose: bool=False
):
    context_nodes = ['c%d' % i for i in range(len(setting_list))]
    context_adjacencies = set(itr.permutations(context_nodes, r=2))
    known_iv_adjacencies = set.union(*(
        {('c%s' % i, node) for node in setting['known_interventions']} for i, setting in enumerate(setting_list)
    ))
    fixed_orders = set(itr.product(context_nodes, range(nnodes)))

    initial_permutations = [context_nodes+random.sample(list(range(nnodes)), nnodes) for _ in range(nruns)]

    return gsp(
        nnodes+len(context_nodes),
        combined_ci_tester,
        initial_permutations=initial_permutations,
        fixed_orders=fixed_orders,
        fixed_adjacencies=context_adjacencies|known_iv_adjacencies,
        verbose=verbose
    )


def gsp(
        nnodes: int,
        ci_tester: CI_Tester,
        depth: Optional[int]=4,
        nruns: int=5,
        verbose: bool=False,
        restricted: bool=True,
        initial_undirected: Optional[Union[str, UndirectedGraph]]='threshold',
        initial_permutations: Optional[List]=None,
        fixed_orders=set(),
        fixed_adjacencies=set(),
        fixed_gaps=set()
) -> (DAG, List[List[Dict]]):
    """
    Use the Greedy Sparsest Permutation (GSP) algorithm to estimate the Markov equivalence class of the data-generating
    DAG.

    Parameters
    ----------
    nnodes:
        Number of nodes in the graph.
    ci_tester:
        A conditional independence tester, which has a method is_ci taking two sets A and B, and a conditioning set C,
        and returns True/False.
    depth:
        Maximum depth in depth-first search. Use None for infinite search depth.
    nruns:
        Number of runs of the algorithm. Each run starts at a random permutation and the sparsest DAG from all
        runs is returned.
    verbose:
        TODO
    initial_undirected:
        Option to find the starting permutation by using the minimum degree algorithm on an undirected graph that is
        Markov to the data. You can provide the undirected graph yourself, use the default 'threshold' to do simple
        thresholding on the partial correlation matrix, or select 'None' to start at a random permutation.
    initial_permutations:
        A list of initial permutations with which to start the algorithm. This option is helpful when there is
        background knowledge on orders. This option is mutually exclusive with initial_undirected.
    fixed_orders:
        Tuples (i, j) where i is known to come before j.
    fixed_adjacencies:
        Tuples (i, j) where i and j are known to be adjacent.
    fixed_gaps:
        Tuples (i, j) where i and j are known to be non-adjacent.

    See Also
    --------
    igsp, unknown_target_igsp

    Return
    ------
    (est_dag, summaries)
    """
    def _reverse_arc(dag, covered_arcs, i, j):
        new_dag = dag.copy()
        parents = dag.parents_of(i)

        new_dag.reverse_arc(i, j)
        if parents:
            for parent in parents:
                rest = parents - {parent}
                i_parent_fixed = (i, parent) in fixed_adjacencies or (parent, i) in fixed_adjacencies
                j_parent_fixed = (j, parent) in fixed_adjacencies or (parent, j) in fixed_adjacencies
                if not i_parent_fixed and ci_tester.is_ci(i, parent, [*rest, j]):
                    new_dag.remove_arc(parent, i)
                if not j_parent_fixed and ci_tester.is_ci(j, parent, [*rest]):
                    new_dag.remove_arc(parent, j)

        new_covered_arcs = covered_arcs.copy() - dag.incident_arcs(i) - dag.incident_arcs(j)
        for k, l in new_dag.incident_arcs(i) | new_dag.incident_arcs(j):
            if new_dag.parents_of(k) == new_dag.parents_of(l) - {k}:
                new_covered_arcs.add((k, l))
        return new_dag, new_covered_arcs

    if initial_permutations is None and isinstance(initial_undirected, str):
        if initial_undirected == 'threshold':
            initial_undirected = threshold_ug(nnodes, ci_tester)
        else:
            raise ValueError("initial_undirected must be one of 'threshold', or an UndirectedGraph")

    summaries = []
    min_dag = None
    for r in range(nruns):
        summary = []
        # === STARTING VALUES
        if initial_permutations is not None:
            starting_perm = initial_permutations[r]
        elif initial_undirected:
            starting_perm = min_degree_alg(initial_undirected, ci_tester)
        else:
            starting_perm = random.sample(list(range(nnodes)), nnodes)
        current_dag = perm2dag(
            starting_perm,
            ci_tester,
            restricted=restricted,
            fixed_adjacencies=fixed_adjacencies,
            fixed_gaps=fixed_gaps
        )
        if verbose: print("=== STARTING DAG:", current_dag)
        current_covered_arcs = current_dag.reversible_arcs() - fixed_orders
        next_dags = [
            _reverse_arc(current_dag, current_covered_arcs, i, j)
            for i, j in current_covered_arcs
        ]

        # === RECORDS FOR DEPTH-FIRST SEARCH
        all_visited_dags = set()
        trace = []

        # === SEARCH!
        while True:
            summary.append({'dag': current_dag, 'depth': len(trace), 'num_arcs': len(current_dag.arcs)})
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
                    _reverse_arc(current_dag, current_covered_arcs, i, j)
                    for i, j in current_covered_arcs
                ]
                next_dags = [(d, cov_arcs) for d, cov_arcs in next_dags if frozenset(d.arcs) not in all_visited_dags]
            else:
                if len(trace) == 0:  # reached minimum within search depth
                    break
                else:  # backtrack
                    current_dag, current_covered_arcs, next_dags = trace.pop()

        # === END OF RUN
        summaries.append(summary)
        if min_dag is None or len(current_dag.arcs) < len(min_dag.arcs):
            min_dag = current_dag

    return min_dag, summaries


def igsp(
        setting_list: List[Dict],
        nnodes: Any,
        ci_tester: CI_Tester,
        invariance_tester: InvarianceTester,
        depth: Optional[int] = 4,
        nruns: int = 5,
        verbose: bool = False,
        starting_permutations = None
):
    only_single_node = all(len(setting['interventions']) <= 1 for setting in setting_list)
    interventions2setting_nums = {
        frozenset(setting['interventions']): setting_num
        for setting_num, setting in enumerate(setting_list)
    }

    def _is_icovered(i, j):
        """
        i -> j is I-covered if:
        1) if {i} is an intervention, then f^{i}(j) = f(j)
        """
        setting_num = interventions2setting_nums.get(frozenset({i}))
        if setting_num is not None and not invariance_tester.is_invariant(j, 0, setting_num):
            return False
        # for iv_nodes in samples.keys():
        #     if j in iv_nodes and i not in iv_nodes:
        #         if not _get_is_variant(iv_nodes, i, None):
        #             return False
        return True

    def _reverse_arc(dag, i, j):
        new_dag = dag.copy()
        parents = dag.parents_of(i)

        new_dag.reverse_arc(i, j)
        if parents:
            for parent in parents:
                rest = parents - {parent}
                if ci_tester.is_ci(i, parent, [*rest, j]):
                    new_dag.remove_arc(parent, i)
                if ci_tester.is_ci(j, parent, cond_set=[*rest]):
                    new_dag.remove_arc(parent, j)

        new_covered_arcs = new_dag.reversible_arcs()
        new_icovered_arcs = [(i, j) for i, j in new_covered_arcs if _is_icovered(i, j)]
        new_contradicting = _get_contradicting_arcs(new_dag)

        return new_dag, new_icovered_arcs, new_contradicting

    def _is_i_contradicting(i, j, dag):
        """
        i -> j is I-contradicting if either:
        1) there exists S, a subset of the neighbors of j besides i, s.t. f^I(j|S) = f(j|S) for all I
            containing i but not j
        2) there exists I with j \in I but i \not\in I, s.t. f^I(i|S) \not\eq f(i|S) for all subsets S
            of the neighbors of i besides j

        If there are only single node interventions, this condition becomes:
        1) {i} \in I and f^{i}(j) = f(j)
        or
        2) {j} \in I and f^{j}(i) \neq f(i)
        """
        if only_single_node:
            setting_num_i = interventions2setting_nums.get(frozenset({i}))
            if setting_num_i is not None and invariance_tester.is_invariant(j, context=setting_num_i):
                return True
            setting_num_j = interventions2setting_nums.get(frozenset({j}))
            if setting_num_j is not None and not invariance_tester.is_invariant(i, context=setting_num_j):
                return True
            return False
        else:
            # === TEST CONDITION 1
            neighbors_j = dag.neighbors_of(j) - {i}
            for s in powerset(neighbors_j):
                for setting_num, setting in enumerate(setting_list):
                    if i in setting['interventions'] and j not in setting['interventions']:
                        if not invariance_tester.is_invariant(j, context=setting_num, cond_set=s):
                            return True

            neighbors_i = dag.neighbors_of(i) - {j}
            for setting_num, setting in enumerate(setting_list):
                if j in setting['interventions'] and i not in setting['interventions']:
                    i_always_varies = all(
                        invariance_tester.is_invariant(i, context=setting_num, cond_set=s) for s in powerset(neighbors_i)
                    )
                    if i_always_varies: return True
            return False

    def _get_contradicting_arcs(dag):
        """
        Count the number of I-contradicting arcs in the DAG dag
        """
        contradicting_arcs = {(i, j) for i, j in dag.arcs if _is_icovered(i, j) and _is_i_contradicting(i, j, dag)}
        return contradicting_arcs

    summaries = []
    # === LIST OF DAGS FOUND BY EACH RUN
    finishing_dags = []

    if starting_permutations is not None:
        nruns = len(starting_permutations)
    # === DO MULTIPLE RUNS
    for r in range(nruns):
        summary = []
        # === STARTING VALUES
        if starting_permutations is None:
            starting_perm = random.sample(list(range(nnodes)), nnodes)
        else:
            starting_perm = starting_permutations[r]
        current_dag = perm2dag(starting_perm, ci_tester)
        if verbose: print("=== STARTING RUN %s/%s" % (r+1, nruns))
        current_covered_arcs = current_dag.reversible_arcs()
        current_icovered_arcs = [(i, j) for i, j in current_covered_arcs if _is_icovered(i, j)]
        current_contradicting = _get_contradicting_arcs(current_dag)
        next_dags = [_reverse_arc(current_dag, i, j) for i, j in current_icovered_arcs]
        random.shuffle(next_dags)

        # === RECORDS FOR DEPTH-FIRST SEARCH
        all_visited_dags = set()
        trace = []
        min_dag_run = (current_dag, current_contradicting)

        # === SEARCH
        while True:
            summary.append({
                'dag': current_dag,
                'num_arcs': len(current_dag.arcs),
                'num_contradicting': len(current_contradicting)
            })
            all_visited_dags.add(frozenset(current_dag.arcs))
            lower_dags = [
                (d, icovered_arcs, contradicting_arcs)
                for d, icovered_arcs, contradicting_arcs in next_dags
                if len(d.arcs) < len(current_dag.arcs)
            ]

            if verbose:
                desc = f'({len(current_dag.arcs)} arcs'
                desc += f', I-covered: {current_icovered_arcs}'
                desc += f', I-contradicting: {current_contradicting})'
                print('-'*len(trace), current_dag, desc)
            if (len(next_dags) > 0 and len(trace) != depth) or len(lower_dags) > 0:
                if len(lower_dags) > 0:  # restart at a lower DAG
                    all_visited_dags = set()
                    trace = []
                    current_dag, current_icovered_arcs, current_contradicting = lower_dags.pop()
                    min_dag_run = (current_dag, current_contradicting)
                    if verbose: print(f"FOUND DAG WITH {len(current_dag.arcs)}) ARCS: {current_dag}")
                else:
                    trace.append((current_dag, current_icovered_arcs, current_contradicting))
                    current_dag, current_icovered_arcs, current_contradicting = next_dags.pop()
                    if len(current_contradicting) < len(min_dag_run[1]):
                        min_dag_run = (current_dag, current_contradicting)
                        if verbose:
                            print(f"FOUND DAG WITH {current_contradicting} CONTRADICTING ARCS: {current_dag}")
                next_dags = [_reverse_arc(current_dag, i, j) for i, j in current_icovered_arcs]
                next_dags = [
                    (d, icovered_arcs, contradicting_arcs)
                    for d, icovered_arcs, contradicting_arcs in next_dags
                    if frozenset(d.arcs) not in all_visited_dags
                ]
                random.shuffle(next_dags)
            # === DEAD END
            else:
                if len(trace) == 0:
                    break
                else:  # len(lower_dags) == 0, len(next_dags) > 0, len(trace) == depth
                    current_dag, current_icovered_arcs, current_contradicting = trace.pop()

        # === END OF RUN
        summaries.append(summary)
        finishing_dags.append(min_dag_run)

    min_dag = min(finishing_dags, key=lambda dag_n: (len(dag_n[0].arcs), len(dag_n[1])))
    # print(min_dag)
    return min_dag[0]


def is_icovered(
        setting_list: List[Dict],
        i: int,
        j: int,
        dag: DAG,
        invariance_tester: InvarianceTester,
):
    """
    Tell if an edge i->j is I-covered with respect to the invariance tests.

    True if, for all I s.t. i \in I, the distribution of j given its parents varies between the observational and
    interventional data.

    setting_list:
        A list of dictionaries that provide meta-information about each setting.
        The first setting must be observational.
    i:
        Source of the edge being tested.
    j:
        Target of the edge being tested.
    """
    parents_j = list(dag.parents_of(j))

    for setting_num, setting in enumerate(setting_list):
        if i in setting['interventions']:
            if invariance_tester.is_invariant(j, context=setting_num, cond_set=parents_j):
                return False

    return True


def unknown_target_igsp(
        setting_list: List[Dict],
        nnodes: int,
        ci_tester: CI_Tester,
        invariance_tester: InvarianceTester,
        depth: Optional[int]=4,
        nruns: int=5,
        verbose: bool=False,
        starting_permutations = None
) -> (DAG, List[Set[int]]):
    """
    Use the Unknown Target Interventional Greedy Sparsest Permutation algorithm to estimate a DAG in the I-MEC of the
    data-generating DAG.

    Parameters
    ----------
    setting_list:
        A list of dictionaries that provide meta-information about each non-observational setting.
    nnodes:
        Number of nodes in the graph.
    ci_tester:
        A conditional independence tester object, which has a method is_ci taking two sets A and B, and a conditioning
        set C, and returns True/False.
    invariance_tester:
        An invariance tester object, which has a method is_invariant taking a node, two settings, and a conditioning
        set C, and returns True/False.
    depth:
        Maximum depth in depth-first search. Use None for infinite search depth.
    nruns:
        Number of runs of the algorithm. Each run starts at a random permutation and the sparsest DAG from all
        runs is returned.
    """
    def _is_icovered(i, j, dag):
        """
        Check if the edge i->j is I-covered in the DAG dag
        """
        parents_j = frozenset(dag.parents_of(j))
        for setting_num, setting in enumerate(setting_list):
            if i in setting['known_interventions']:
                if invariance_tester.is_invariant(j, context=setting_num, cond_set=parents_j):
                    return False
        return True

    def _get_variants(dag):
        """
        Count the number of variances for the DAG dag
        """
        variants = set()

        for i in dag.nodes:
            parents_i = frozenset(dag.parents_of(i))
            for setting_num, setting in enumerate(setting_list):
                if not invariance_tester.is_invariant(i, context=setting_num, cond_set=parents_i):
                    variants.add((setting_num, i, parents_i))

        return variants

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
                if ci_tester.is_ci(i, parent, [*rest, j]):
                    new_dag.remove_arc(parent, i)
                if ci_tester.is_ci(j, parent, cond_set=[*rest]):
                    new_dag.remove_arc(parent, j)

        # new_i_covered_arcs = i_covered_arcs.copy() - dag.incident_arcs(i) - dag.incident_arcs(j)
        # for k, l in new_dag.incident_arcs(i) | new_dag.incident_arcs(j):
        #     if new_dag.parents_of(k) == new_dag.parents_of(l) - {k} and _is_icovered(i, j, dag):
        #         new_i_covered_arcs.add((k, l))

        new_covered_arcs = new_dag.reversible_arcs()
        new_i_covered_arcs = [(i, j) for i, j in new_covered_arcs if _is_icovered(i, j, new_dag)]
        variants = _get_variants(new_dag)
        new_score = len(new_dag.arcs) + len(variants)
        intervention_targets = [set() for _ in range(len(setting_list))]
        for setting_num, i, parents_i in variants:
            intervention_targets[setting_num].add(i)

        return new_dag, new_i_covered_arcs, new_score, intervention_targets

    # === MINIMUM DAG AND SCORE FOUND BY ANY RUN
    min_dag = None
    min_score = float('inf')
    learned_intervention_targets = None

    if starting_permutations is not None:
        nruns = len(starting_permutations)
    # === MULTIPLE RUNS
    for r in range(nruns):
        # === STARTING VALUES
        if starting_permutations is not None:
            starting_perm = starting_permutations[r]
        else:
            starting_perm = random.sample(list(range(nnodes)), nnodes)
        current_dag = perm2dag(starting_perm, ci_tester)
        variants = _get_variants(current_dag)
        current_intervention_targets = [set() for _ in range(len(setting_list))]
        for setting_num, i, parents_i in variants:
            current_intervention_targets[setting_num].add(i)
        current_score = len(current_dag.arcs) + len(variants)
        if verbose: print("=== STARTING DAG:", current_dag, "== SCORE:", current_score)

        current_covered_arcs = current_dag.reversible_arcs()
        current_i_covered_arcs = [(i, j) for i, j in current_covered_arcs if _is_icovered(i, j, current_dag)]
        if verbose: print("=== STARTING I-COVERED ARCS:", current_i_covered_arcs)
        next_dags = [_reverse_arc_igsp(current_dag, current_i_covered_arcs, i, j) for i, j in current_i_covered_arcs]
        next_dags = [
            (d, i_cov_arcs, score, iv_targets) for d, i_cov_arcs, score, iv_targets in next_dags
            if score <= current_score
        ]
        random.shuffle(next_dags)

        # === RECORDS FOR DEPTH-FIRST SEARCH
        all_visited_dags = set()
        trace = []

        # === SEARCH!
        while True:
            if verbose:
                print('-'*len(trace), current_dag, '(%d arcs)' % len(current_dag.arcs), 'I-covered arcs:', current_i_covered_arcs, 'score:', current_score)
            all_visited_dags.add(frozenset(current_dag.arcs))
            lower_dags = [
                (d, i_cov_arcs, score, iv_targets) for d, i_cov_arcs, score, iv_targets in next_dags
                if score < current_score
            ]

            if (len(next_dags) > 0 and len(trace) != depth) or len(lower_dags) > 0:
                if len(lower_dags) > 0:  # restart at a lower DAG
                    all_visited_dags = set()
                    trace = []
                    current_dag, current_i_covered_arcs, current_score, current_intervention_targets = lower_dags.pop()
                    if verbose: print("FOUND DAG WITH LOWER SCORE:", current_dag, "== SCORE:", current_score)
                else:
                    trace.append((current_dag, current_i_covered_arcs, next_dags, current_intervention_targets))
                    current_dag, current_i_covered_arcs, current_score, current_intervention_targets = next_dags.pop()
                next_dags = [
                    _reverse_arc_igsp(current_dag, current_i_covered_arcs, i, j)
                    for i, j in current_i_covered_arcs
                ]
                next_dags = [
                    (d, i_cov_arcs, score, iv_targets) for d, i_cov_arcs, score, iv_targets in next_dags
                    if score <= current_score
                ]
                next_dags = [
                    (d, i_cov_arcs, score, iv_targets) for d, i_cov_arcs, score, iv_targets in next_dags
                    if frozenset(d.arcs) not in all_visited_dags
                ]
                random.shuffle(next_dags)
            # === DEAD END ===
            else:
                if len(trace) == 0:  # reached minimum within search depth
                    break
                else:  # backtrack
                    current_dag, current_i_covered_arcs, next_dags, current_intervention_targets = trace.pop()

        if min_dag is None or current_score < min_score:
            min_dag = current_dag
            min_score = current_score
            learned_intervention_targets = current_intervention_targets
        if verbose: print("=== FINISHED RUN %s/%s ===" % (r+1, nruns))

    return min_dag, learned_intervention_targets








