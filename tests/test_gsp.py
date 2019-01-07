from unittest import TestCase
import unittest
import numpy as np
import causaldag as cd
from causaldag.inference.structural import gsp, perm2dag, is_icovered
from causaldag.utils.ci_tests import kci_test_vector, ki_test_vector, gauss_ci_test
import random


class TestDAG(TestCase):
    def test_gsp(self):
        ndags = 100
        nnodes = 8
        nsamples = 1000
        nneighbors_list = list(range(1, 8))
        for nneighbors in nneighbors_list:
            dags = cd.rand.directed_erdos(nnodes, nneighbors/(nnodes-1), ndags)
            gdags = [cd.rand.rand_weights(dag) for dag in dags]
            samples_by_dag = [gdag.sample(nsamples) for gdag in gdags]
            corr_by_dag = [np.corrcoef(samples, rowvar=False) for samples in samples_by_dag]
            est_dags = [gsp(dict(C=corr, n=nsamples), random.sample(list(range(nnodes)), nnodes), gauss_ci_test) for corr in corr_by_dag]

    # def test_perm2dag(self):
    #     gdag = cd.GaussDAG([0, 1, 2], arcs={(0, 1): 5, (0, 2): 5})
    #     samples = gdag.sample(100)
    #     est_dag = perm2dag(samples, [2, 0, 1], kci_test, ki_test)
    #     print(est_dag)
    #
    # def test_isicovered(self):
    #     d = cd.GaussDAG([0, 1], arcs={(0, 1)})
    #
    #     samples = {}
    #     samples[frozenset()] = d.sample(100)
    #     samples[frozenset({1})] = d.sample_interventional({1: cd.GaussIntervention(0, 5)}, 1000)
    #
    #     is_icov = is_icovered(samples, 1, 0, kci_test)
    #     print(is_icov)

    # def test_3nodes(self):
    #     gdag = cd.GaussDAG([1, 2, 3], arcs={(1, 2), (2, 3)})
    #     samples = gdag.sample_interventional({1: cd.ConstantIntervention(0)})
    #     estimated_perm, estimated_imap = igsp(samples, [3, 2, 1])


if __name__ == '__main__':
    unittest.main()
