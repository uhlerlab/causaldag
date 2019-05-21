import itertools as itr
from ...classes import UndirectedGraph
from ...utils.ci_tests import CI_Tester


def threshold_ug(nodes: set, ci_tester: CI_Tester) -> UndirectedGraph:
    edges = {(i, j) for i, j in itr.combinations(nodes, 2) if not ci_tester.is_ci(i, j, nodes - {i, j})}
    return UndirectedGraph(nodes, edges)
