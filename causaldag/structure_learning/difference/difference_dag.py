from causaldag.utils.core_utils import powerset
from causaldag.utils.ci_tests import CI_Tester
from causaldag.utils.invariance_tests import InvarianceTester


def dci_skeleton(
        nodes: set,
        invariance_tester: InvarianceTester,
        difference_ug: set,
        max_set_size: int=None
):
    skeleton = difference_ug.copy()
    for i, j in difference_ug:
        for cond_set in powerset(nodes - {i, j}, r_max=max_set_size):
            if invariance_tester.is_invariant(j, 0, {*cond_set, i}):
                skeleton -= frozenset({i, j})
            elif invariance_tester.is_invariant(i, 0, {*cond_set, j}):
                skeleton -= frozenset({i, j})
    return skeleton


def dci(
        nodes: set,
        ci_tester: CI_Tester,
        invariance_tester: InvarianceTester,
        difference_ug: set,
        max_set_size: int = None
):
    skeleton = dci_skeleton(nodes, invariance_tester, difference_ug, max_set_size=max_set_size)


def dci_orient(ci_tester):
    pass
