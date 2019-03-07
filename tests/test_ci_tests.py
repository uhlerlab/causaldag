from unittest import TestCase
import unittest
import numpy as np
import causaldag as cd
from causaldag import GaussIntervention, ScalingIntervention
import os
import random

# np.random.seed(1728)
# random.seed(1728)


class TestKCI(TestCase):
    # def test_dependent(self):
    #     Y = np.random.multivariate_normal([0, 0], np.eye(2), 500).round(4)
    #     E = Y
    #     X = np.random.multivariate_normal([0, 0], np.eye(2), 500).round(4)
    #     # np.savetxt(os.path.expanduser('~/Desktop/Y.txt'), Y)
    #     # np.savetxt(os.path.expanduser('~/Desktop/E.txt'), E)
    #     # np.savetxt(os.path.expanduser('~/Desktop/X.txt'), X)
    #     statistic, critval, pval = cd.utils.ci_tests.kci(Y, E, X)
    #     print(statistic, critval, pval)
    #
    #     self.assertTrue(pval < .05)

    # def test_independent(self):
    #     num_trials = 100
    #     nsamples = 200
    #
    #     alpha = .05
    #     false_negatives = 0
    #     for _ in range(num_trials):
    #         Y = np.random.multivariate_normal([0, 0], np.eye(2), nsamples).round(4)
    #         X = np.random.multivariate_normal([0, 0], np.eye(2), nsamples).round(4)
    #         np.savetxt(os.path.expanduser('~/Desktop/Y.txt'), Y)
    #         np.savetxt(os.path.expanduser('~/Desktop/X.txt'), X)
    #         test_results = cd.utils.ci_tests.kci_invariance_test(Y, X, 0, alpha=alpha)
    #
    #         if test_results['p_value'] < alpha:
    #             false_negatives += 1
    #
    #     print("Expected number of false negatives:", alpha*num_trials)
    #     print("Actual number of false negatives:", false_negatives)

    # def test_ki_independent(self):
    #     false_negatives = 0
    #     alpha = .05
    #     num_tests = 100
    #     for i in range(num_tests):
    #         X = np.random.multivariate_normal([0], np.eye(1), 500).round(4)
    #         Y = np.random.multivariate_normal([0, 0], np.eye(2), 500).round(4)
    #         statistic, critval, pval = cd.utils.ci_tests.ki_test(X, Y)
    #         if pval < .05:
    #             false_negatives += 1
    #     self.assertTrue(0 < false_negatives < num_tests*alpha*2)

    # def test_kci_invariance_no_cond_set_false_negatives(self):
    #     false_negatives = 0
    #     alpha = .05
    #     num_tests = 100
    #     nsamples = 100
    #     for i in range(num_tests):
    #         d = cd.GaussDAG(nodes=[0, 1, 2], arcs={(0, 1), (0, 2), (1, 2)})
    #         samples = d.sample(nsamples)
    #         iv_samples = d.sample_interventional_perfect({1: GaussIntervention(mean=0, variance=1)}, nsamples=nsamples)
    #         print(np.cov(samples.T), np.cov(iv_samples.T))
    #         test_results = cd.utils.ci_tests.kci_invariance_test(samples, iv_samples, 2, alpha=alpha)
    #         print(test_results)
    #         if not test_results['reject']:  # should be rejecting the hypothesis of invariance
    #             false_negatives += 1
    #     print("Number of false negatives:", false_negatives)

    # def test_hsic_invariance_no_cond_set_false_negatives(self):
    #     false_negatives = 0
    #     alpha = .05
    #     num_tests = 100
    #     nsamples = 200
    #     for i in range(num_tests):
    #         d = cd.GaussDAG(nodes=[0, 1, 2], arcs={(0, 1), (0, 2), (1, 2)})
    #         samples = d.sample(nsamples)
    #         iv_samples = d.sample_interventional_perfect({1: GaussIntervention(mean=0, variance=1)},
    #                                                      nsamples=nsamples)
    #         # print(np.cov(samples.T), np.cov(iv_samples.T))
    #         test_results = cd.utils.ci_tests.hsic_invariance_test(samples, iv_samples, 2, alpha=alpha)
    #         print(test_results)
    #         if not test_results['reject']:  # should be rejecting the hypothesis of invariance
    #             false_negatives += 1
    #     print("Number of false negatives:", false_negatives)

    # def test_hsic_invariance_no_cond_set_false_positives(self):
    #     false_positives = 0
    #     alpha = .05
    #     num_tests = 100
    #     nsamples = 200
    #     for i in range(num_tests):
    #         d = cd.GaussDAG(nodes=[0, 1, 2], arcs={(0, 1), (0, 2), (1, 2)})
    #         samples = d.sample(nsamples)
    #         iv_samples = d.sample_interventional_perfect({1: GaussIntervention(mean=0, variance=1)},
    #                                                      nsamples=nsamples)
    #         # print(np.cov(samples.T), np.cov(iv_samples.T))
    #         test_results = cd.utils.ci_tests.hsic_invariance_test(samples, iv_samples, 0, alpha=alpha)
    #         print(test_results)
    #         if test_results['reject']:  # should be accepting the hypothesis of invariance
    #             false_positives += 1
    #     print("Number of false positives:", false_positives)
    #     print("Expected number of false positives:", alpha*num_tests)

    # def test_hsic_invariance_cond_set_false_positives(self):
    #     false_positives = 0
    #     alpha = .05
    #     num_tests = 100
    #     nsamples = 200
    #     for i in range(num_tests):
    #         d = cd.GaussDAG(nodes=[0, 1, 2], arcs={(0, 1), (0, 2), (1, 2)})
    #         samples = d.sample(nsamples)
    #         iv_samples = d.sample_interventional_soft({1: ScalingIntervention(0)},
    #                                                      nsamples=nsamples)
    #         test_results = cd.utils.ci_tests.hsic_invariance_test(samples, iv_samples, 2, cond_set=[0,1], alpha=alpha)
    #         if test_results['reject']:  # should be accepting the hypothesis of invariance
    #             false_positives += 1
    #     print("Number of false positives:", false_positives)
    #     print("Expected number of false positives:", alpha * num_tests)

    def test_hsic_invariance_cond_set_false_negatives(self):
        false_negatives = 0
        alpha = .05
        num_tests = 100
        nsamples = 200
        for i in range(num_tests):
            d = cd.GaussDAG(nodes=[0, 1, 2], arcs={(0, 1), (0, 2), (1, 2)})
            samples = d.sample(nsamples)
            iv_samples = d.sample_interventional_soft({1: ScalingIntervention(0)}, nsamples=nsamples)
            test_results = cd.utils.ci_tests.hsic_invariance_test(samples, iv_samples, 0, cond_set=[1],
                                                                  alpha=alpha)
            if not test_results['reject']:  # should be rejecting the hypothesis of invariance
                false_negatives += 1
        print("Number of false negatives:", false_negatives)

    # def test_ki_invariance_no_cond_set_false_positives(self):
    #     false_positives = 0
    #     alpha = .05
    #     num_tests = 100
    #     nsamples = 500
    #     for i in range(num_tests):
    #         d = cd.GaussDAG(nodes=[0, 1], arcs={(0, 1)})
    #         samples = d.sample(nsamples)
    #         iv_samples = d.sample_interventional({frozenset({1}): cd.GaussIntervention(10, 1)}, nsamples=nsamples)
    #         test_results = cd.utils.ci_tests.kci_invariance_test(samples, iv_samples, 0)
    #         print(test_results)
    #         if test_results['reject']:
    #             false_positives += 1
    #     print("Number of false positives:", false_positives)
    #
    # def test_kci_no_cond_set_false_negatives(self):
    #     false_negatives = 0
    #     alpha = .05
    #     num_tests = 100
    #     nsamples = 500
    #     for i in range(num_tests):
    #         d = cd.GaussDAG(nodes=[0, 1], arcs={(0, 1)})
    #         samples = d.sample(nsamples)
    #         test_results = cd.utils.ci_tests.kci_test(samples, 0, 1)
    #         print(test_results)
    #         if not test_results['reject']:
    #             false_negatives += 1
    #     print("Number of false negatives:", false_negatives)

    # def test_kci_conditionally_independent(self):
    #     ntrials = 100
    #     alpha = .05
    #     g = cd.GaussDAG([0, 1, 2], arcs={(0, 1), (0, 2)})
    #
    #     false_negatives = 0
    #     for i in range(ntrials):
    #         samples = g.sample(500)
    #         test_results = cd.utils.ci_tests.kci_test(samples, 1, 2, 0, alpha=alpha)
    #         if test_results['reject']: false_negatives += 1
    #     print(false_negatives)

    # def test_gauss_ci_independent(self):
    #     false_negatives = 0
    #     num_tests = 100
    #     n_samples = 500
    #     alpha = 0.05
    #     for i in range(num_tests):
    #         samples = np.random.multivariate_normal([1, 3], np.eye(2), n_samples).round(4)
    #         corr = np.corrcoef(samples, rowvar=False)
    #         results = cd.utils.ci_tests.gauss_ci_test(dict(n=n_samples, C=corr), 0, 1, alpha=alpha)
    #         if results['reject']:
    #             false_negatives += 1
    #
    #     print("Expected number of false negatives:", alpha*num_tests)
    #     print("Number of false negatives:", false_negatives)
    #
    # def test_gauss_ci(self):
    #     false_negatives = 0
    #     num_tests = 100
    #     n_samples = 500
    #     alpha = 0.05
    #     d = cd.GaussDAG([0, 1, 2], arcs={(0, 1), (0, 2)})
    #     for i in range(num_tests):
    #         corr = np.corrcoef(d.sample(n_samples), rowvar=False)
    #         results = cd.utils.ci_tests.gauss_ci_test(dict(n=n_samples, C=corr), 1, 2, cond_set=[0], alpha=alpha)
    #         if results['reject']:
    #             false_negatives += 1
    #
    #     false_positives = 0
    #     for i in range(num_tests):
    #         corr = np.corrcoef(d.sample(n_samples), rowvar=False)
    #         results = cd.utils.ci_tests.gauss_ci_test(dict(n=n_samples, C=corr), 1, 2, alpha=alpha)
    #         if not results['reject']:
    #             false_positives += 1
    #
    #     print("Expected number of false negatives:", alpha*num_tests)
    #     print("Number of false negatives:", false_negatives)
    #
    #     print("Number of false positives (should be low):", false_positives)


if __name__ == '__main__':
    unittest.main()
