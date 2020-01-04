from numpy import ix_
from numpy.linalg import inv


class RegressionHelper:
    def __init__(self, suffstat):
        self.suffstat = suffstat
        self.S = suffstat['S']
        self.P = suffstat['P']
        self.p = self.suffstat['S'].shape[0]
        self.n = self.suffstat['n']

    def regression(self, i: int, c: list=None):
        S, P = self.S, self.P
        d = [j for j in range(self.p) if j not in c and j != i] + [i]

        if c is None or len(c) == 0:
            coefs = []
            var = S[i, i]
            S_inv = None

        # use Schur complement when conditioning to keep inverted submatrix small
        elif len(c) < self.p / 2:
            S_inv = inv(S[ix_(c, c)])
            coefs = S_inv @ S[c, i]
            var = S[i, i] - S[i, c] @ S_inv @ S[c, i]

        # use Schur complement when marginalizing to keep inverted submatrix small
        else:
            P_inv = inv(P[ix_(d, d)])
            S_inv = P[ix_(c, c)] - P[ix_(c, d)] @ P_inv @ P[ix_(d, c)]
            coefs = S_inv @ S[c, i]
            var = inv(P[ix_(d, d)])[-1, -1]

        # correct the variance to account for the number of degrees of freedom
        var = var * self.n / (self.n - len(c) + 1)

        return coefs, var, S_inv


if __name__ == '__main__':
    from causaldag.rand.graphs import directed_erdos, rand_weights
    from causaldag.utils.ci_tests import gauss_ci_suffstat
    from causaldag.utils.core_utils import powerset
    from sklearn.linear_model import LinearRegression
    import numpy as np
    from tqdm import trange

    nnodes = 10
    nodes = set(range(nnodes))
    exp_nbrs = 2
    nsamples = 100
    dag = directed_erdos(nnodes, exp_nbrs/(nnodes-1))
    gdag = rand_weights(dag)
    samples = gdag.sample(nsamples)
    suff = gauss_ci_suffstat(samples)
    reg_helper = RegressionHelper(suff)
    lr = LinearRegression()

    for i in trange(nnodes):
        for c in powerset(nodes - {i}, r_min=1):
            c = list(c)
            coefs, var = reg_helper.regression(i, c)
            lr.fit(samples[:, c], samples[:, i])
            var2 = np.var(samples[:, c] @ coefs - samples[:, i], ddof=len(c))
            if not np.isclose(coefs, lr.coef_).all():
                print(coefs, lr.coef_)
            if not np.isclose(var, var2):
                print(var, var2)

