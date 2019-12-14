import itertools as itr
from causaldag.utils.ci_tests import CI_Tester


def estimate_adjacent(nodes: set, target_node, ci_tester: CI_Tester):
    nnodes = len(nodes)
    nbrs = nodes - {target_node}
    for c_size in range(nnodes-1):
        nbrs_ = nbrs.copy()
        for nbr in nbrs_:
            if len(nbrs_-{nbr}) >= c_size:
                for cond_set in itr.combinations(nbrs_-{nbr}, c_size):
                    if ci_tester.is_ci(target_node, nbr, cond_set):
                        nbrs.remove(nbr)
                        break
    return nbrs


if __name__ == '__main__':
    import causaldag as cd
    from causaldag.utils.ci_tests import MemoizedCI_Tester, dsep_test
    import numpy as np
    import random
    random.seed(129318290)
    np.random.seed(129318290)

    nnodes = 6
    nodes = set(range(nnodes))
    exp_nbrs = 2
    ngraphs = 100
    target_nodes = [random.randint(0, nnodes) for _ in range(ngraphs)]
    dags = cd.rand.directed_erdos(nnodes, exp_nbrs/(nnodes-1), ngraphs)
    ci_testers = [MemoizedCI_Tester(dsep_test, d) for d in dags]
    est_adjacent_list = [estimate_adjacent(nodes, target_node, ci_tester) for target_node, ci_tester in zip(target_nodes, ci_testers)]
    true_adjacent_list = [d.neighbors_of(target_node) for d, target_node in zip(dags, target_nodes)]
    match = np.array([est_adjacent == true_adjacent for est_adjacent, true_adjacent in zip(est_adjacent_list, true_adjacent_list)])
    print(np.where(~match))
    print([est_adjacent - true_adjacent for est_adjacent, true_adjacent in zip(est_adjacent_list, true_adjacent_list)])
    print([true_adjacent - true_adjacent for est_adjacent, true_adjacent in zip(est_adjacent_list, true_adjacent_list)])

    ix = 1
    print(dags[ix])
    print(target_nodes[ix])
    print(est_adjacent_list[ix])


