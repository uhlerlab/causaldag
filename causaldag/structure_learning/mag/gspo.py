import causaldag as cd
from causaldag.structure_learning.dag import perm2dag, min_degree_alg_amat, gsp
from causaldag.structure_learning.undirected import threshold_ug
import random
import itertools as itr


def apply_lmc(imap, i, j):
    if imap.has_directed(i, j):
        imap.remove_directed(i, j)
        imap.add_bidirected(i, j)
    else:
        imap.remove_bidirected(i, j)
        imap.add_directed(i, j)


def get_lmc_altered_edges(imap: cd.AncestralGraph, i, j, ci_tester):
    """
    Given an IMAP and a legitimate mark change applied to it, test which edges can be removed after the legitimate
    mark change.
    """
    imap_copy = imap.copy()
    apply_lmc(imap_copy, i, j)

    desc_j = imap.descendants_of(j) | {j}

    for k, l in list(imap.directed) + list(imap.bidirected):
        if k in desc_j or l in desc_j:
            c = imap_copy.ancestors_of({k, l}) - {k, l}
            if ci_tester.is_ci(k, l, c):
                imap_copy.remove_edge(k, l)
    removed_dir = imap.directed - imap_copy.directed
    if imap.has_directed(i, j): removed_dir = removed_dir - {(i, j)}
    removed_bidir = bidirected_frozenset(imap) - bidirected_frozenset(imap_copy)
    if imap.has_bidirected(i, j): removed_bidir = removed_bidir - {frozenset({i, j})}

    return removed_dir, removed_bidir


def bidirected_frozenset(m):
    return frozenset({frozenset({*e}) for e in m._bidirected})


def gspo(
        nodes: set,
        ci_tester,
        depth=4,
        initial_imap='permutation',
        strict=True,
        verbose=False,
        max_iters=float('inf'),
        nruns=5,
):
    """
    Estimate a MAG using the Greedy Sparsest Poset algorithm.

    Parameters
    ----------
    nodes:
        Labels of nodes in the graph.
    ci_tester:
        A conditional independence tester, which has a method is_ci taking two sets A and B, and a conditioning set C,
        and returns True/False.
    depth:
        Maximum depth in depth-first search. Use None for infinite search depth.
    initial_imap:
        String indicating how to obtain the initial IMAP. Must be "permutation" or "empty".
    strict:
        If True, check discriminating paths condition for legitimate mark changes.
    verbose:
        If True, print information about algorithm progress.
    max_iters:
        Maximum number of depth-first search steps without score improvement before stopping.
    nruns:
        Number of times to run the algorithm (each run may vary due to randomness in tie-breaking and/or starting
        imap.

    Return
    ------
    An estimated MAG
    """
    if initial_imap == 'permutation':
        ug = threshold_ug(nodes, ci_tester)
        amat = ug.to_amat()
        perms = [min_degree_alg_amat(amat) for _ in range(nruns)]
        dags = [perm2dag(perm, ci_tester) for perm in perms]
        starting_imaps = [cd.AncestralGraph(dag.nodes, directed=dag.arcs) for dag in dags]
    elif initial_imap == 'empty':
        edges = {(i, j) for i, j in itr.combinations(nodes, 2) if not ci_tester.is_ci(i, j)}
        starting_imaps = [cd.AncestralGraph(nodes, bidirected=edges) for _ in range(nruns)]
    elif initial_imap == 'gsp':
        ug = threshold_ug(nodes, ci_tester)
        amat = ug.to_amat()
        perms = [min_degree_alg_amat(amat) for _ in range(nruns)]
        dags = [gsp(nodes, ci_tester, nruns=1, initial_permutations=[perm]) for perm in perms]
        starting_imaps = [cd.AncestralGraph(dag.nodes, directed=dag.arcs) for dag, _ in dags]

    get_alt_edges = get_lmc_altered_edges

    sparsest_imap = None
    for r in range(nruns):
        current_imap = starting_imaps[r]
        if verbose: print(f"Starting run {r} with {current_imap.num_edges} edges")

        # TODO: BOTTLENECK
        current_lmcs_directed, current_lmcs_bidirected = current_imap.legitimate_mark_changes(strict=strict)
        current_lmcs = current_lmcs_directed | current_lmcs_bidirected

        # TODO: BOTTLENECK
        lmcs2altered_edges = [
            (lmc, get_alt_edges(current_imap, *lmc, ci_tester))
            for lmc in current_lmcs
        ]
        lmcs2altered_edges = [(lmc, (a, b)) for lmc, (a, b) in lmcs2altered_edges if a is not None]
        lmcs2edge_delta = [
            (lmc, len(removed_dir) + len(removed_bidir))
            for lmc, (removed_dir, removed_bidir) in lmcs2altered_edges
        ]

        mag2number = dict()
        graph_counter = 0
        trace = []
        iters_since_improvement = 0
        while True:
            if iters_since_improvement > max_iters:
                break

            mag_hash = (frozenset(current_imap._directed), bidirected_frozenset(current_imap))
            if mag_hash not in mag2number:
                mag2number[mag_hash] = graph_counter
            graph_num = mag2number[mag_hash]
            if verbose: print(f"Number of visited MAGs: {len(mag2number)}. Exploring MAG #{graph_num} with {current_imap.num_edges} edges.")
            max_delta = max([delta for lmc, delta in lmcs2edge_delta], default=0)

            sparser_exists = max_delta > 0
            keep_searching_mec = len(trace) != depth and len(lmcs2altered_edges) > 0

            if sparser_exists:
                trace = []

                lmc_ix = random.choice([ix for ix, (lmc, delta) in enumerate(lmcs2edge_delta) if delta == max_delta])
                (i, j), (removed_dir, removed_bidir) = lmcs2altered_edges.pop(lmc_ix)
                apply_lmc(current_imap, i, j)
                current_imap.remove_edges(removed_dir | removed_bidir)

                if verbose: print(f"Starting over at a sparser IMAP with {current_imap.num_edges} edges")
            elif keep_searching_mec:
                if verbose: print(f"{'='*len(trace)}Continuing search through the MEC at {current_imap.num_edges} edges. "
                                  f"Picking from {len(lmcs2altered_edges)} neighbors of #{graph_num}.")
                trace.append((current_imap.copy(), current_lmcs, lmcs2altered_edges, lmcs2edge_delta))
                (i, j), _ = lmcs2altered_edges.pop(0)
                lmcs2edge_delta.pop(0)
                apply_lmc(current_imap, i, j)
            elif len(trace) != 0:  # BACKTRACK IF POSSIBLE
                if verbose: print(f"{'='*len(trace)}Backtracking")
                current_imap, current_lmcs, lmcs2altered_edges, lmcs2edge_delta = trace.pop()
                iters_since_improvement += 1
            else:
                break

            # IF WE MOVED TO A NOVEL IMAP, WE NEED TO UPDATE LMCs
            if sparser_exists or keep_searching_mec:
                graph_counter += 1
                current_lmcs_dir, current_lmcs_bidir = current_imap.legitimate_mark_changes(strict=strict)
                current_lmcs = current_lmcs_dir | current_lmcs_bidir
                lmcs2altered_edges = [
                    (lmc, get_alt_edges(current_imap, *lmc, ci_tester))
                    for lmc in current_lmcs
                ]
                lmcs2altered_edges = [(lmc, (a, b)) for lmc, (a, b) in lmcs2altered_edges if a is not None]
                current_directed, current_bidirected = frozenset(current_imap.directed), bidirected_frozenset(current_imap)

                # === FILTER OUT ALREADY-VISITED IMAPS
                filtered_lmcs2altered_edges = []
                for lmc, (removed_dir, removed_bidir) in lmcs2altered_edges:
                    if current_imap.has_directed(*lmc):
                        new_directed = current_directed - {lmc} - removed_dir
                        new_bidirected = current_bidirected | {frozenset({*lmc})} - {frozenset({*e}) for e in removed_bidir}
                    else:
                        new_directed = current_directed | {lmc} - removed_dir
                        new_bidirected = current_bidirected - {frozenset({*lmc})} - {frozenset({*e}) for e in removed_bidir}

                    if (new_directed, new_bidirected) not in mag2number:
                        filtered_lmcs2altered_edges.append((lmc, (removed_dir, removed_bidir)))
                lmcs2altered_edges = filtered_lmcs2altered_edges

                lmcs2edge_delta = [
                    (lmc, len(removed_dir)+len(removed_bidir))
                    for lmc, (removed_dir, removed_bidir) in lmcs2altered_edges
                ]
        if sparsest_imap is None or sparsest_imap.num_edges > current_imap.num_edges:
            sparsest_imap = current_imap

    return current_imap
