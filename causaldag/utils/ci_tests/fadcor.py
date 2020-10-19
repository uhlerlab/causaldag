import numpy as np
import ipdb
from typing import Union, List, Dict
from causaldag.utils.ci_tests._utils import residuals
from causaldag.utils.core_utils import to_list


def order_vector(x) -> np.ndarray:
    """
    Return vector of ranks for each element of x. For example, [1.1, 2.2, 0.5] -> [1, 2, 0]
    """
    return np.argsort(np.argsort(x))


def fadcor_test(
        suffstat: np.ndarray,
        i: int,
        j: int,
        cond_set: Union[List[int], int]=None,
        thresh: float=1.,
        verbose=False
) -> Dict:
    """
    Test for (conditional) independence using Distance Covariance. If a conditioning set is
    specified, first perform non-parametric regression, then test residuals.

    Parameters
    ----------
    suffstat:
        Matrix of samples.
    i:
        column position of first variable.
    j:
        column position of second variable.
    cond_set:
        column positions of conditioning set.
    alpha:
        Significance level of the test.

    Returns
    -------

    """
    cond_set = to_list(cond_set)
    if len(cond_set) == 0:
        return fadcor_test_vector(suffstat[:, i], suffstat[:, j], thresh=thresh, verbose=verbose)
    else:
        if verbose: print("Computing residuals")
        residuals_i, residuals_j = residuals(suffstat, i, j, cond_set)
        return fadcor_test_vector(residuals_i, residuals_j, thresh=thresh, verbose=verbose)


def fadcor_test_vector(x: np.ndarray, y: np.ndarray, verbose=False, thresh=1.):
    """
    Test for independence of X and Y using Fast Computing for Distance Covariance (FaDCor).

    Parameters
    ----------
    x:
        vector of samples from X.
    y:
        vector of samples from Y.

    Returns
    -------

    """
    n = len(x)

    # STEP 1
    if verbose: print("Ordering vectors")
    ranks_x, ranks_y = order_vector(x), order_vector(y)

    # STEP 2: COMPUTE PARTIAL SUMS OF ORDER STATISTICS
    if verbose: print("computing cumulative sums")
    x_sort_ixs, y_sort_ixs = np.argsort(x), np.argsort(y)
    s_x = np.cumsum(x[x_sort_ixs]) - x[x_sort_ixs]
    s_y = np.cumsum(y[y_sort_ixs]) - y[y_sort_ixs]

    # STEP 3: COMPUTE BETA
    if verbose: print("computing beta")
    beta_x = s_x[ranks_x]
    beta_y = s_y[ranks_y]

    # STEP 4/5: COMPUTE A AND B
    if verbose: print("Computing a. and b.")
    a_dot = x.sum() + (2*ranks_x - n) * x - 2*beta_x
    b_dot = y.sum() + (2*ranks_y - n) * y - 2*beta_y

    # STEP 6: COMPUTE AVERAGE DISTANCES WITHIN X AND Y DATASETS
    if verbose: print("Computing a.. and b..")
    a_dotdot = 2 * ranks_x.T @ x - 2 * beta_x.sum()
    b_dotdot = 2 * ranks_y.T @ y - 2 * beta_y.sum()

    # STEP 7
    if verbose: print("Computing gammas")
    gamma1 = partial_sum2d(x, y, np.ones(n))
    gamma_xy = partial_sum2d(x, y, x*y)
    gamma_y = partial_sum2d(x, y, y)
    gamma_x = partial_sum2d(x, y, x)

    # STEP 8
    if verbose: print("Computing LOO sums")
    loo_sums = x * y * gamma1 + gamma_xy - x * gamma_y - y * gamma_x
    loo_sum = loo_sums.sum()

    # STEP 9
    omega = 1/(n*(n-3)) * loo_sum - 2/(n*(n-2)*(n-3)) * a_dot.T @ b_dot + a_dotdot*b_dotdot/(n*(n-1)*(n-2)*(n-3))

    return dict(
        statistic=omega,
        reject=omega > thresh,
    )


def partial_sum2d(x: np.ndarray, y: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Return \gamma, with \gamma_i = \sum_{j \neq i} c_j S_{ij} for S_{ij} = sign((x_i - x_j)*(y_i - y_j)).
    """
    # STEP 1
    x_ixs = np.argsort(x)
    rank_x = order_vector(x)
    x, y, c = x[x_ixs], y[x_ixs], c[x_ixs]

    # STEP 2
    rank_y = order_vector(y)
    y_ixs = np.argsort(y)

    # STEP 3
    c_sorted = c[y_ixs]
    s_y = np.cumsum(c_sorted) - c_sorted
    s_y = s_y[rank_y]
    # s_y2 = [sum([c[j] for j in range(len(y)) if y[j] < y[i]]) for i in range(len(y))]
    # print('--')
    # print(s_y)
    # print(s_y2)
    # print('---')

    # STEP 4
    s_x = np.cumsum(c) - c

    # STEP 5
    c_dot = c.sum()

    # STEP 6
    d = dyad_update(order_vector(y), c)

    # STEP 7
    gamma = c_dot - c - 2*s_y - 2*s_x + 4 * d

    return gamma[rank_x]


def dyad_update(y, c) -> np.ndarray:
    """
    Return \gamma, with \gamma_i = \sum_{j < i, y_j < y_i} c_j
    """
    y = np.array(y)

    # STEP 1
    n = len(y)
    L = int(np.ceil(np.log2(n)))

    # STEP 2
    S = np.zeros([L, n])
    gamma = np.zeros(n)

    for i in range(1, n):
        # (3a)
        rows = np.arange(L)
        cols = (np.floor(y[i-1]) / 2**rows).astype(int)
        S[rows, cols] += c[i-1]

        # (3b)
        ells = get_ells(y[i])
        if len(ells) > 0:
            ks = 2**ells
            ks = np.cumsum(ks)
            ks = ks * 2.**-ells
            ks = (ks - 1).astype(int)

            # === REPLACED WITH ABOVE FOR SPEED
            # ks = np.zeros(len(ells))
            # for j in range(1, len(ells)):
            #     ks[j] = np.sum(2**ells[m] for m in range(j)) * 2.**(-ells[j])
            # ks = ks.astype(int)

            # (3c)
            gamma[i] = S[ells, ks].sum()

    return gamma


def get_ells(y) -> np.ndarray:
    """
    Return vector ell of length L s.t. 2**l[1] + ... + 2**l[L] = y, sorted in descending order
    """
    b = bin(y)[2:]
    length = len(b)
    ells = [(length - 1 - i) for i, val in enumerate(b) if val == '1']
    ells = np.array(ells)
    return ells


# for debugging only
def _dyad_update_simple(y, c):
    n = len(y)
    gamma = np.zeros(n)
    for i in range(n):
        gamma[i] = sum([c[j] for j in range(i) if y[j] < y[i]])
    return gamma


# for debugging only
def _partial_sum_simple(x, y, c):
    n = len(x)

    r = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                s_ij = np.sign((x[i]- x[j]) * (y[i] - y[j]))
                r[i] += c[j] * s_ij
    return r


if __name__ == '__main__':
    import causaldag as cd
    from tqdm import tqdm
    import random

    np.random.seed(1818)
    random.seed(11)

    nsamples = 10000
    ntrials = 10
    perms = [np.random.permutation(list(range(nsamples))) for _ in range(ntrials)]
    samples_list = [np.random.normal(size=nsamples) for _ in range(ntrials)]
    d = cd.GaussDAG(nodes=[0, 1], arcs=set())
    samples = d.sample(1000)

    TEST_TIME = False
    if TEST_TIME:
        samples_list = [d.sample(nsamples) for _ in range(ntrials)]
        for samples in tqdm(samples_list):
            fadcor_test_vector(samples[:, 0], samples[:, 1])

    TEST_PARTIAL_SUM = False
    if TEST_PARTIAL_SUM:
        x = samples[:, 0]
        y = samples[:, 1]
        c = np.random.normal(size=len(x))
        res1 = partial_sum2d(x, y, c)
        res2 = _partial_sum_simple(x, y, c)
        print('res1:', res1)
        print('res2:', res2)
        print(np.isclose(res1, res2).all())

    TEST_CORRECTNESS = False
    if TEST_CORRECTNESS:
        om = fadcor_test_vector(samples[:, 0], samples[:, 1])
        np.save('test.npy', samples)


    PROFILE_DYAD = False
    if PROFILE_DYAD:
        from line_profiler import LineProfiler

        def run_dyad_update():
            for perm, samples in tqdm(zip(perms, samples_list), total=ntrials):
                d1 = dyad_update(perm, samples)

        profiler = LineProfiler()
        profiler.add_function(dyad_update)
        profiler.runcall(run_dyad_update)
        profiler.print_stats()

    PROFILE_FADCOR = True
    if PROFILE_FADCOR:
        from line_profiler import LineProfiler
        samples_list = [d.sample(nsamples) for _ in range(ntrials)]

        def run_fadcor():
            for samples in tqdm(samples_list):
                d1 = fadcor_test_vector(samples[:, 0], samples[:, 1])


        profiler = LineProfiler()
        profiler.add_function(fadcor_test_vector)
        profiler.add_function(partial_sum2d)
        profiler.add_function(dyad_update)
        profiler.runcall(run_fadcor)
        profiler.print_stats()

    COMPARE_TIME = False
    if COMPARE_TIME:
        for perm, samples in tqdm(zip(perms, samples_list), total=ntrials):
            d1 = dyad_update(perm, samples)
        for perm, samples in tqdm(zip(perms, samples_list), total=ntrials):
            d2 = _dyad_update_simple(perm, samples)

    COMPARE_ANSWER = False
    if COMPARE_ANSWER:
        for perm, samples in tqdm(zip(perms, samples_list), total=ntrials):
            d1 = dyad_update(perm, samples)
            d2 = _dyad_update_simple(perm, samples)
            ne = np.where(~np.isclose(d1, d2))[0]
            if len(ne) > 0:
                ipdb.set_trace()
    # o = fadcor_test_vector(samples[:, 0], samples[:, 1])


