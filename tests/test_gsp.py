from unittest import TestCase
import unittest
import numpy as np
import causaldag as cd
from causaldag.inference.structural import gsp, perm2dag, is_icovered
from causaldag.utils.ci_tests import kci_test, ki_test


class TestDAG(TestCase):
    def test_gsp(self):
        gdag = cd.GaussDAG([0, 1, 2, 3], arcs={(0, 1), (1, 2), (1, 3)})
        samples = gdag.sample(600)
        est_dag = gsp(samples, [0, 2, 1, 3], kci_test, ki_test, verbose=True)
        print(est_dag)

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
