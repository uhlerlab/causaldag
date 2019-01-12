from unittest import TestCase
import unittest
import numpy as np
import causaldag as cd
from scipy.stats import multivariate_normal


class TestGaussDAG(TestCase):
    def setUp(self):
        w = np.zeros((3, 3))
        w[0, 1] = 1
        w[0, 2] = -1
        w[1, 2] = 4
        self.gdag = cd.GaussDAG.from_amat(w)

    def test_arcs(self):
        self.assertEqual(self.gdag.arcs, {(0, 1), (0, 2), (1, 2)})

    def test_add_node(self):
        pass

    def test_add_arc(self):
        pass

    def test_remove_node(self):
        pass

    def test_remove_arc(self):
        pass

    # def test_from_precision(self):
    #     prec = np.array([
    #         [14, 13, -3],
    #         [13, 26, -5],
    #         [-3, -5, 1]
    #     ])
    #     order1 = [0, 1, 2]
    #     gdag1 = cd.GaussDAG.from_precision(prec, order1)
    #     self.assertDictEqual(gdag1.arc_weights, {(1, 2): 2, (1, 3): 3, (2, 3): 5})

    def test_logpdf_observational(self):
        amat = np.array([
            [0, 2, 3],
            [0, 0, 5],
            [0, 0, 0]
        ])
        gdag = cd.GaussDAG.from_amat(amat)
        samples = gdag.sample(100)
        logpdf_gdag = gdag.logpdf(samples)
        logpdf_scipy = multivariate_normal.logpdf(samples, cov=gdag.covariance)
        self.assertTrue(all(np.isclose(logpdf_gdag, logpdf_scipy)))


if __name__ == '__main__':
    unittest.main()
