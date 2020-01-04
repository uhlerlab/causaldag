from causaldag.utils.core_utils import powerset
from causaldag.utils.regression import RegressionHelper
from scipy.special import ncfdtr
from numpy.linalg import inv


def dci(
        nodes: set,
        difference_ug: set,
        suffstat1: dict,
        suffstat2: dict,
        alpha_skeleton: float=.01,
        alpha_orient: float=.01,
        max_set_size: int = None
):
    """
    Use the Difference Causal Inference (DCI) algorithm to estimate the difference-DAG between two settings.

    Parameters
    ----------
    nodes:
        Labels of nodes in the graph.
    suffstat1:
        Dictionary of sufficient statistics for the first dataset.
    suffstat2:
        Dictionary of sufficient statistics for the second dataset.
    alpha_skeleton:
        todo
    alpha_orient:
        todo
    difference_ug:
        Estimated set of edges in the difference UG.
    max_set_size:
        Maximum conditioning set size used to test regression invariance.

    See Also
    --------
    dci_skeleton, dci_orient

    Returns
    -------

    """
    rh1 = RegressionHelper(suffstat1)
    rh2 = RegressionHelper(suffstat2)
    skeleton = dci_skeleton(nodes, difference_ug, max_set_size=max_set_size, rh1=rh1, rh2=rh2, alpha=alpha_skeleton)
    return dci_orient(skeleton, max_set_size=max_set_size, alpha=alpha_orient, rh1=rh1, rh2=rh2)


def dci_skeleton(
        nodes: set,
        difference_ug: set,
        suffstat1: dict=None,
        suffstat2: dict=None,
        alpha: float=.05,
        max_set_size: int=None,
        rh1: RegressionHelper=None,
        rh2: RegressionHelper=None,
):
    """
    Perform Phase I of the Difference Causal Inference (DCI) algorithm, i.e., estimate the skeleton of the
    difference DAG.

    Parameters
    ----------
    nodes:
        Labels of nodes in the graph.
    difference_ug:
        Estimated set of edges in the difference UG.
    suffstat1:
        Dictionary of sufficient statistics for the first dataset.
    suffstat2:
        Dictionary of sufficient statistics for the second dataset.
    alpha:
        todo
    max_set_size:
        Maximum conditioning set size used to test regression invariance.
    rh1:
        todo
    rh2:
        todo

    See Also
    --------
    dci, dci_orient

    Returns
    -------

    """
    if rh1 is None:
        rh1 = RegressionHelper(suffstat1)
        rh2 = RegressionHelper(suffstat2)

    n1 = rh1.suffstat['n']
    n2 = rh2.suffstat['n']
    skeleton = {frozenset({i, j}) for i, j in difference_ug}
    for i, j in difference_ug:
        for cond_set in powerset(nodes - {i, j}, r_max=max_set_size):
            cond_set_i, cond_set_j = [*cond_set, j], [*cond_set, i]

            # calculate regression coefficients (j regressed on cond_set_j) for both datasets
            beta1_i, var1_i, precision1 = rh1.regression(i, cond_set_i)
            beta2_i, var2_i, precision2 = rh2.regression(i, cond_set_i)

            # compute statistic and p-value
            j_ix = cond_set_i.index(j)
            stat_i = (beta1_i[j_ix] - beta2_i[j_ix])**2 * inv(var1_i*precision1/(n1-1) + var2_i*precision2/(n2-1))[j_ix, j_ix]
            pval_i = ncfdtr(1, n1 + n2 - len(cond_set_i) - len(cond_set_j), 0, stat_i)
            pval_i = 2 * min(pval_i, 1 - pval_i)

            #  remove i-j from skeleton if i regressed on (j, cond_set) is invariant
            i_invariant = pval_i > alpha
            if i_invariant:
                skeleton.remove(frozenset({i, j}))
                break

            # calculate regression coefficients (i regressed on cond_set_i) for both datasets
            beta1_j, var1_j, precision1 = rh1.regression(j, cond_set_j)
            beta2_j, var2_j, precision2 = rh2.regression(j, cond_set_j)

            # compute statistic and p-value
            i_ix = cond_set_j.index(i)
            stat_j = (beta1_j[i_ix] - beta2_j[i_ix]) ** 2 * inv(var1_j*precision1/(n1-1) + var2_j*precision2/(n2-1))[i_ix, i_ix]
            pval_j = 1 - ncfdtr(1, n1 + n2 - len(cond_set_i) - len(cond_set_j), 0, stat_j)
            pval_j = 2 * min(pval_j, 1 - pval_j)

            #  remove i-j from skeleton if j regressed on (i, cond_set) is invariant
            j_invariant = pval_j > alpha
            if j_invariant:
                skeleton.remove(frozenset({i, j}))
                break

    return skeleton


def dci_orient(
        skeleton: set,
        suffstat1: dict = None,
        suffstat2: dict = None,
        alpha: float = .05,
        max_set_size: int = None,
        rh1: RegressionHelper=None,
        rh2: RegressionHelper=None,
):
    """
    Perform Phase I of the Difference Causal Inference (DCI) algorithm, i.e., orient edges in the skeleton of the
    difference DAG.

    Parameters
    ----------
    skeleton:
        The estimated skeleton of the difference DAG.
    suffstat1:
        Dictionary of sufficient statistics for the first dataset.
    suffstat2:
        Dictionary of sufficient statistics for the second dataset.
    alpha:
        todo
    max_set_size:
        Maximum conditioning set size used to test regression invariance.
    rh1:
        todo
    rh2:
        todo

    See Also
    --------
    dci, dci_skeleton

    Returns
    -------

    """
    if rh1 is None:
        rh1 = RegressionHelper(suffstat1)
        rh2 = RegressionHelper(suffstat2)

    nodes = {i for i, j in skeleton} | {j for i, j in skeleton}
    oriented_edges = set()

    n1 = rh1.suffstat['n']
    n2 = rh2.suffstat['n']
    for i, j in skeleton:
        for cond_i, cond_j in zip(powerset(nodes - {i}, r_max=max_set_size), powerset(nodes - {j}, r_max=max_set_size)):
            # compute residual variances for i
            beta1_i, var1_i, _ = rh1.regression(i, cond_i)
            beta2_i, var2_i, _ = rh2.regression(i, cond_i)
            # check pvalue
            pvalue_i = ncfdtr(n1 - len(cond_i), n2 - len(cond_i), 0, var1_i/var2_i)
            pvalue_i = min(pvalue_i, 1-pvalue_i)
            if pvalue_i > alpha:
                oriented_edges.add((j, i) if j in cond_i else (i, j))
                break

            # compute residual variances for j
            beta1_j, var1_j, _ = rh1.regression(i, cond_j)
            beta2_j, var2_j, _ = rh2.regression(i, cond_j)
            # check pvalue
            pvalue_j = ncfdtr(n1 - len(cond_j), n2 - len(cond_j), 0, var1_j / var2_j)
            pvalue_j = min(pvalue_j, 1 - pvalue_j)
            if pvalue_j > alpha:
                oriented_edges.add((i, j) if i in cond_j else (j, i))
                break

    unoriented_edges = skeleton - {frozenset({i, j}) for i, j in oriented_edges}
    return oriented_edges, unoriented_edges
    

if __name__ == '__main__':
    from causaldag.rand.graphs import directed_erdos, rand_weights
    import causaldag as cd
    from causaldag.utils.ci_tests import gauss_ci_suffstat
    import itertools as itr

    import sys
    import os
    sys.path.append(os.path.join(os.path.expanduser('~'), 'dropbox', 'learning differences of DAGs', 'py_work'))
    from algs.skeleton.yc_skeleton import infer_skeleton
    from algs.orienting.yc_orienting import infer_directions

    nnodes = 10
    nodes_ = set(range(nnodes))
    exp_nbrs = 2
    nsamples1 = 10000
    nsamples2 = 10100
    alpha_ = .05
    candidate_edges = set(itr.combinations(nodes_, 2))

    d = directed_erdos(nnodes, exp_nbrs/(nnodes-1))
    g1 = rand_weights(d)
    amat1 = g1.to_amat()
    amat2 = amat1.copy()
    amat2[0, 1] += 10
    g2 = cd.GaussDAG.from_amat(amat2)

    samples1 = g1.sample(nsamples1)
    samples2 = g2.sample(nsamples2)
    suff1 = gauss_ci_suffstat(samples1)
    suff2 = gauss_ci_suffstat(samples2)

    # skel = dci_skeleton(nodes_, candidate_edges, suff1, suff2, alpha=alpha_)
    # print(len(skel))

    # skel2, _, _, _ = infer_skeleton(samples1, samples2, candidate_edges, changed_nodes=nodes_, alpha=alpha_)
    skel = {frozenset({0, 1})}
    oriented_edges_, unoriented_edges_ = dci_orient(skel, suff1, suff2)
    oriented_edges2 = infer_directions(samples1, samples2, skel, changed_nodes=nodes_, alpha=alpha_)
    print(oriented_edges_, unoriented_edges_)
    print(oriented_edges2)



