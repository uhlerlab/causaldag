from typing import Dict, Optional, Any, List, Set, Union
from causaldag import DAG
import itertools as itr
from conditional_independence import CI_Tester
from causaldag.classes.custom_types import UndirectedEdge
from causaldag.utils.invariance_tests import InvarianceTester
from causaldag.utils.core_utils import powerset, iszero
import random
from causaldag.structure_learning.undirected import threshold_ug, partial_correlation_threshold
from causaldag import UndirectedGraph
import numpy as np
from tqdm import trange, tqdm
from causaldag.utils.core_utils import powerset
from math import factorial
from conditional_independence import MemoizedCI_Tester, get_ci_tester


from causaldag.structure_learning.dag import permutation2dag, sparsest_permutation, gsp


def perm2dag_new(
        perm: list,
        samples,
        memoize=True,
        ci_test="partial_correlation",
        verbose=False,
        fixed_adjacencies: Set[UndirectedEdge]=set(),
        fixed_gaps: Set[UndirectedEdge]=set(),
        progress=False,
        **kwargs,
):
    """
    Given a permutation, find the minimal IMAP consistent with that permutation and the results of conditional independence
    tests from ci_tester.

    Parameters
    ----------
    perm:
        list of nodes representing the permutation.
    verbose:
        if True, log each CI test.
    fixed_adjacencies:
        set of nodes known to be adjacent.
    fixed_gaps:
        set of nodes known not to be adjacent.

    Examples
    --------
    >>> from causaldag.utils.ci_tests import MemoizedCI_Tester, gauss_ci_test, gauss_ci_suffstat
    >>> perm = [0,1,2]
    >>> suffstat = gauss_ci_suffstat(samples)
    >>> ci_tester = MemoizedCI_Tester(gauss_ci_test, suffstat)
    >>> permutation2dag(perm, ci_tester, fixed_gaps={frozenset({1, 2})})
    """
    ci_tester = get_ci_tester(samples, ci_test, memoize, **kwargs)

    return permutation2dag(
        perm,
        ci_tester,
        verbose=verbose,
        fixed_adjacencies=fixed_adjacencies,
        fixed_gaps=fixed_gaps,
        progress=progress
    )


def sparsest_permutation_new(
        nodes,
        samples,
        memoize=True,
        ci_test="partial_correlation",
        progress=False,
        **kwargs
):
    """
    Run the Sparsest Permutation algorithm, finding the sparsest minimal IMAP associated with any permutation.

    Parameters
    ----------
    nodes:
        list of nodes.
    progress:
        if True, show a progress bar over the enumeration of permutations.

    Examples
    --------
    >>> from causaldag.utils.ci_tests import MemoizedCI_Tester, gauss_ci_test, gauss_ci_suffstat
    >>> import causaldag as cd
    >>> import random
    >>> import numpy as np
    >>> random.seed(1212)
    >>> np.random.seed(12131)
    >>> nnodes = 7
    >>> d = cd.rand.directed_erdos(nnodes, exp_nbrs=2)
    >>> g = cd.rand.rand_weights(d)
    >>> samples = g.sample(1000)
    >>> suffstat = gauss_ci_suffstat(samples)
    >>> ci_tester = MemoizedCI_Tester(gauss_ci_test, suffstat, alpha=1e-3)
    >>> est_dag = cd.sparsest_permutation(set(range(nnodes)), ci_tester, progress=True)
    >>> true_cpdag = d.cpdag()
    >>> est_cpdag = est_dag.cpdag()
    >>> print(true_cpdag.shd(est_cpdag))
    >>> 0
    """
    ci_tester = get_ci_tester(samples, ci_test, memoize, **kwargs)
    return sparsest_permutation(nodes, ci_tester, progress=progress)


def gsp_new(
        nodes: set,
        samples,
        memoize=True,
        ci_test="partial_correlation",
        depth: Optional[int] = 4,
        nruns: int = 5,
        verbose: bool = False,
        initial_undirected: Optional[Union[str, UndirectedGraph]] = 'threshold',
        initial_permutations: Optional[List] = None,
        fixed_orders=set(),
        fixed_adjacencies=set(),
        fixed_gaps=set(),
        use_lowest=True,
        max_iters=float('inf'),
        factor=2,
        progress_bar=False,
        summarize=False,
        **kwargs
) -> (DAG, List[List[Dict]]):
    """
    Use the Greedy Sparsest Permutation (GSP) algorithm to estimate the Markov equivalence class of the data-generating
    DAG.

    Parameters
    ----------
    nodes:
        Labels of nodes in the graph.
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
    pcalg, igsp, unknown_target_igsp

    Return
    ------
    (est_dag, summaries)
    """
    ci_tester = get_ci_tester(samples, test=ci_test, memoize=memoize, **kwargs)
    return gsp(
        nodes,
        ci_tester,
        depth,
        nruns,
        verbose,
        initial_undirected,
        initial_permutations,
        fixed_orders,
        fixed_adjacencies,
        fixed_gaps,
        use_lowest=use_lowest,
        max_iters=max_iters,
        factor=factor,
        progress_bar=progress_bar,
        summarize=summarize
    )
