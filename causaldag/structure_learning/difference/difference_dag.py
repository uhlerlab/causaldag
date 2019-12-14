from causaldag.utils.core_utils import powerset
from causaldag.utils.ci_tests import CI_Tester
from causaldag.utils.invariance_tests import InvarianceTester


def dci(
        nodes: set,
        invariance_tester: InvarianceTester,
        difference_ug: set,
        max_set_size: int = None
):
    """
    Use the Difference Causal Inference (DCI) algorithm to estimate the difference-DAG between two settings.

    Parameters
    ----------
    nodes:
        Labels of nodes in the graph.
    invariance_tester
    difference_ug:

    max_set_size:
        Maximum conditioning set size used to test regression invariance.

    See Also
    --------
    dci_skeleton, dci_orient

    Returns
    -------

    """
    skeleton = dci_skeleton(nodes, invariance_tester, difference_ug, max_set_size=max_set_size)
    return dci_orient(skeleton, max_set_size=max_set_size)


def dci_skeleton(
        nodes: set,
        invariance_tester: InvarianceTester,
        difference_ug: set,
        max_set_size: int=None
):
    """
    Perform Phase I of the Difference Causal Inference (DCI) algorithm, i.e., estimate the skeleton of the
    difference DAG.

    Parameters
    ----------
    nodes:
        Labels of nodes in the graph.
    invariance_tester:
        An invariance tester, which has a method is_invariant taking a node i and a conditioning set C,
        and returning whether or not the distribution of i|C is the same between the two contexts.
    difference_ug
    max_set_size:
        Maximum conditioning set size used to test regression invariance.

    See Also
    --------
    dci, dci_orient

    Returns
    -------

    """
    skeleton = difference_ug.copy()
    for i, j in difference_ug:
        for cond_set in powerset(nodes - {i, j}, r_max=max_set_size):
            j_invariant = invariance_tester.is_invariant(j, 0, {*cond_set, i})
            i_invariant = invariance_tester.is_invariant(i, 0, {*cond_set, j})
            if j_invariant or i_invariant:
                skeleton -= frozenset({i, j})
                break
    return skeleton


def dci_orient(
        skeleton: set,
        max_set_size: int = None
):
    """
    Perform Phase I of the Difference Causal Inference (DCI) algorithm, i.e., orient edges in the skeleton of the
    difference DAG.

    Parameters
    ----------
    skeleton:
        The estimated skeleton of the difference DAG.
    max_set_size:
        Maximum conditioning set size used to test regression invariance.

    See Also
    --------
    dci, dci_skeleton

    Returns
    -------

    """
    nodes = {i for i, j in skeleton} | {j for i, j in skeleton}
    for i, j in skeleton:
        for cond_i, cond_j in zip(powerset(nodes - {i}, r_max=max_set_size), powerset(nodes - {j}, r_max=max_set_size)):
            # TODO
            pass
