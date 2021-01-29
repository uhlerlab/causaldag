import numpy as np
import numba

@numba.jit
def numba_inv(A):
    return np.linalg.inv(A)


def local_gaussian_bic_score(node, parents, suffstat, lambda_=None):
    n = suffstat["n"]
    lambda_ = lambda_ if lambda_ is not None else -.5 * np.log(n)
    C, i, p = suffstat['S'], node, list(parents)
    C *= (n-1)/n
    var = C[i, i] if not p else C[i, i] - C[i, p] @ numba_inv(C[np.ix_(p, p)]) @ C[p, i]
    log_prob = -.5*n*(1 + np.log(2*np.pi*var))
    penalty_term = lambda_*(2 + len(parents))
    # print("log_prob: ", log_prob)
    # print("penalty_term", penalty_term)

    return log_prob + penalty_term

def local_gaussian_interventional_bic_score(node, parents, suffstat_dict, lambda_=None):
    # TODO: Account for bias and sample means, maybe replace /N-1 with /N
    parents = list(parents)
    parents_and_node = parents + [node]
    p = len(parents)
    C = np.zeros((p+1, p+1))
    total_num_samples = 0

    for intervened_nodes, suffstat in suffstat_dict.items():
        if node not in intervened_nodes:
            C += suffstat['S'][np.ix_(parents_and_node, parents_and_node)] * (suffstat['n'] - 1)
            total_num_samples += suffstat['n']

    if total_num_samples == 0:
        return 0

    n = total_num_samples
    C /= n # TODO: fix
    lambda_ = lambda_ if lambda_ is not None else -.5 * np.log(n)
    var = C[-1, -1] if not p else C[-1, -1] - C[-1, :-1] @ numba_inv(C[:-1, :-1]) @ C[:-1, -1]
    log_prob = -.5*n*(1 + np.log(2*np.pi*var))
    penalty_term = lambda_*(2 + len(parents))

    return log_prob + penalty_term


