import numpy as np
from causaldag.utils.core_utils import random_max
from tqdm import trange


def resit(
        samples: np.ndarray,
        regression_function,  # todo: hyperparameters should be CV'd
        dependence_function,
        progress: bool = False
):
    nsamples, nnodes = samples.shape
    nodes = set(range(nnodes))
    perm = []

    r = trange if progress else range
    for _ in r(nnodes):
        node_dependences = dict()
        for node in nodes:
            other_nodes = list(nodes - {node})
            residuals = regression_function(samples[:, node], samples[:, other_nodes])
            dependence = dependence_function(residuals, samples[:, other_nodes])
            node_dependences[node] = dependence
        print(node_dependences)
        weakest_node = random_max(node_dependences, minimize=True)
        nodes -= {weakest_node}
        perm = [weakest_node, *perm]
        print(perm)

    return perm


if __name__ == '__main__':
    from numpy.linalg import inv
    from causaldag.utils.ci_tests import hsic_test_vector
    from causaldag.rand.graphs import rand_additive_basis, directed_erdos
    from scipy.special import expit
    import pygam


    def identity(x: np.ndarray):
        return x


    def square(x: np.ndarray):
        return x**2 + x - 1


    def cubic(x: np.ndarray):
        return x**3


    def logistic(x: np.ndarray):
        return expit(x)


    def linear_reg(target_samples: np.ndarray, cond_samples: np.ndarray):
        # target_samples: n*1
        # cond_samples: n*p
        cond_samples = cond_samples - cond_samples.mean(axis=0)
        target_samples = target_samples - target_samples.mean()
        cond_cov = cond_samples.T @ cond_samples
        coefs = inv(cond_cov) @ cond_samples.T @ target_samples
        residuals = target_samples - cond_samples @ coefs
        # todo: sample splitting for coefficients and residuals
        return residuals


    def gam_reg(target_samples, cond_samples):
        g = pygam.GAM()
        g.fit(cond_samples, target_samples)
        print(g.coef_.shape)
        residuals = g.deviance_residuals(cond_samples, target_samples)
        return residuals


    def hsic_dependence(samples1, samples2):
        res = hsic_test_vector(samples1, samples2)  # TODO: SHOULD CROSS-VALIDATE HYPERPARAMS
        return -res['statistic']


    d = directed_erdos(10, exp_nbrs=9, random_order=False)
    nsamples = 1000
    basis = [cubic]
    cam_dag = rand_additive_basis(d, basis, snr_dict=.9)
    s = cam_dag.sample(nsamples)
    perm = resit(s, gam_reg, hsic_dependence, progress=True)
