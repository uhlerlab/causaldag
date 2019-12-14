from causaldag.classes import UndirectedGraph
import itertools as itr
import numpy as np
from numpy.linalg import inv
from numpy import sqrt, diag, eye
from causaldag.utils import core_utils
from numpy.random import multivariate_normal


class GGM(UndirectedGraph):
    def __init__(self, precision_matrix: np.ndarray, means: np.ndarray=None):
        nnodes = precision_matrix.shape[0]
        self.precision_matrix = precision_matrix
        self.means = means if means is not None else np.ones(nnodes)
        nodes = set(range(nnodes))
        edges = {(i, j) for i, j in itr.combinations(nodes, 2) if precision_matrix[i, j] != 0}
        super().__init__(set(nodes), edges)

        self._covariance = None
        self._correlation = None

    @classmethod
    def from_covariance(cls, covariance: np.ndarray, means: np.ndarray=None):
        g = GGM(inv(covariance), means)
        g._covariance = covariance
        return g

    @classmethod
    def from_adjacency(cls, adjacency_matrix: np.ndarray, variances: np.ndarray=None, means: np.ndarray=None):
        nnodes = adjacency_matrix.shape[0]
        variances = variances if variances is not None else np.ones(nnodes)
        means = means if means is not None else np.zeros(nnodes)

        id_ = eye(nnodes)
        a = adjacency_matrix
        if (variances == 1).all():
            precision = (id_ - a) @ (id_ - a).T
        else:
            precision = (id_ - a) @ diag(variances ** -1) @ (id_ - a).T

        return GGM(precision, means)

    def _ensure_covariance(self):
        if self._covariance is None:
            self._covariance = inv(self.precision_matrix)

    def _ensure_correlation(self):
        if self._correlation is None:
            S = self.covariance
            self._correlation = S / sqrt(diag(S)) / sqrt(diag(S)).reshape([-1, 1])

    @property
    def covariance(self):
        self._ensure_covariance()
        return self._covariance

    @property
    def correlation(self):
        self._ensure_correlation()
        return self._correlation

    def partial_correlation(self, i, j, cond_set):
        """
        Return the partial correlation of i and j conditioned on `cond_set`.

        Parameters
        ----------
        i:
        j:
        cond_set:

        Examples
        --------
        TODO
        """
        cond_set = core_utils.to_set(cond_set)
        if len(cond_set) == 0:
            return self.correlation[i, j]
        else:
            theta = inv(self.correlation[np.ix_([i, j, *cond_set], [i, j, *cond_set])])
            return -theta[0, 1] / np.sqrt(theta[0, 0] * theta[1, 1])

    def sample(self, nsamples: int=1) -> np.ndarray:
        return multivariate_normal(self.means, self.covariance, nsamples)

    def to_gauss_dag(self, perm):
        """
        Return a GaussDAG with the same mean and covariance as this GGM, and is a minimal IMAP of this GGM
        consistent with the node ordering `perm`.

        Parameters
        ----------
        perm:
            The desired permutation, or total order, of the nodes in the result.

        Returns
        -------

        Examples
        --------
        TODO
        """
        from causaldag import DAG, GaussDAG

        d = DAG(nodes=self.nodes)
        ixs = list(itr.chain.from_iterable(((f, s) for f in range(s)) for s in range(len(perm))))
        for i, j in ixs:
            pi_i, pi_j = perm[i], perm[j]
            if not np.isclose(self.partial_correlation(pi_i, pi_j, d.markov_blanket(pi_i)), 0):
                d.add_arc(pi_i, pi_j, unsafe=True)

        arcs = dict()
        means = []
        Sigma = self.covariance
        variances = []
        for i in perm:
            ps = list(d.parents_of(i))

            # === LINEAR REGRESSION TO FIND EDGE WEIGHTS
            S_xx = Sigma[np.ix_(ps, ps)]
            S_xy = Sigma[ps, i]
            coeffs = inv(S_xx) @ S_xy

            # === COMPUTE MEAN AND VARIANCE
            mean = self.means[i] - self.means[ps] @ coeffs.T
            variance = Sigma[i, i] - Sigma[i, ps] @ coeffs

            for p, coeff in zip(ps, coeffs):
                print(p, i)
                arcs[(p, i)] = coeff
            means.append(mean)
            variances.append(variance)

        return GaussDAG(list(range(self.num_nodes)), arcs, means=means, variances=variances)


if __name__ == '__main__':
    P = np.eye(3)
    P[0, 1] = P[1, 0] = .1
    P[2, 1] = P[1, 2] = .1
    g = GGM(P)
    samples = g.sample(1000)
    cov_sample = np.cov(samples, rowvar=False)

    from causaldag import GaussDAG
    A = np.array([[0, .1, 0], [0, 0, .1], [0, 0, 0]])
    gdag = GaussDAG.from_amat(A)
    ggm = GGM.from_covariance(gdag.covariance)
    gdag2 = ggm.to_gauss_dag([0, 1, 2])
    print(gdag.weight_mat)
    print(gdag2.weight_mat)


