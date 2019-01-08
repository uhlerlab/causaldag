from unittest import TestCase
import unittest
import numpy as np
import causaldag as cd
from causaldag.inference.structural import gsp, perm2dag, is_icovered
from causaldag.utils.ci_tests import kci_test_vector, ki_test_vector, gauss_ci_test, kci_invariance_test
import random


class TestDAG(TestCase):
    # def test_gsp(self):
    #     ndags = 100
    #     nnodes = 8
    #     nsamples = 500
    #     nneighbors_list = list(range(1, 8))
    #
    #     mean_shds = []
    #     percent_consistent = []
    #     for nneighbors in nneighbors_list:
    #         print('=== nneighbors = %s ===' % nneighbors)
    #         print("generating DAGs")
    #         dags = cd.rand.directed_erdos(nnodes, nneighbors/(nnodes-1), ndags)
    #         print("generating weights")
    #         gdags = [cd.rand.rand_weights(dag) for dag in dags]
    #         print("generating samples")
    #         samples_by_dag = [gdag.sample(nsamples) for gdag in gdags]
    #         print("computing correlation matrices")
    #         corr_by_dag = [np.corrcoef(samples, rowvar=False) for samples in samples_by_dag]
    #         print("running GSP")
    #         est_dags = [
    #             gsp(dict(C=corr, n=nsamples), nnodes, gauss_ci_test, depth=4, nruns=10, alpha=.01)
    #             for corr in corr_by_dag
    #         ]
    #         # print([str(d) for d in est_dags])
    #         shd_by_dag = np.array([est_dag.shd_skeleton(dag) for est_dag, dag in zip(est_dags, dags)])
    #         match_by_dag = np.sum(shd_by_dag == 0)
    #         mean_shds.append(shd_by_dag.mean())
    #         percent_consistent.append(match_by_dag/ndags)
    #     print("Mean SHDs:", mean_shds)
    #     print("Percent consistent:", percent_consistent)

    def test_utigsp(self):
        ndags = 100
        nnodes = 8
        nsamples = 500
        nneighbors_list = list(range(1, 8))

        mean_shds = []
        percent_consistent = []
        for nneighbors in nneighbors_list:
            print('=== nneighbors = %s ===' % nneighbors)
            print("generating DAGs")
            dags = cd.rand.directed_erdos(nnodes, nneighbors / (nnodes - 1), ndags)
            print("generating weights")
            gdags = [cd.rand.rand_weights(dag) for dag in dags]
            print("generating samples")
            samples_by_dag = []
            for gdag in gdags:
                iv_nodes = frozenset({random.randint(0, nnodes)})
                samples_by_dag.append({
                    frozenset(): gdag.sample(nsamples),
                    iv_nodes: gdag.sample_interventional({iv_nodes: cd.GaussIntervention(0, 2)})
                })
            print("computing correlation matrices")
            corr_by_dag = [np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_by_dag]
            print("running GSP")
            est_dags = [
                gsp(dict(C=corr, n=nsamples), nnodes, gauss_ci_test, depth=4, nruns=10, alpha=.01)
                for corr in corr_by_dag
            ]
            # print([str(d) for d in est_dags])
            shd_by_dag = np.array([est_dag.shd_skeleton(dag) for est_dag, dag in zip(est_dags, dags)])
            match_by_dag = np.sum(shd_by_dag == 0)
            mean_shds.append(shd_by_dag.mean())
            percent_consistent.append(match_by_dag / ndags)
        print("Mean SHDs:", mean_shds)
        print("Percent consistent:", percent_consistent)

    # def test_perm2dag(self):
    #     gdag = cd.GaussDAG([0, 1, 2], arcs={(0, 1): 5, (0, 2): 5})
    #     samples = gdag.sample(100)
    #     est_dag = perm2dag(samples, [2, 0, 1], kci_test, ki_test)
    #     print(est_dag)
    #
    # def test_isicovered(self):
    #     num_trials = 100
    #     alpha = .05
    #     num_samples = 100
    #
    #     false_negatives = 0
    #     for i in range(num_trials):
    #         d = cd.GaussDAG([0, 1], arcs={(0, 1)})
    #
    #         samples = {}
    #         samples[frozenset()] = d.sample(num_samples)
    #         samples[frozenset({1})] = d.sample_interventional({1: cd.GaussIntervention(2, 1)}, num_samples)
    #
    #         d_rev = cd.DAG(arcs={(1, 0)})
    #         is_icov = is_icovered(samples, 1, 0, d_rev, kci_invariance_test, alpha=alpha)
    #
    #         if not is_icov:
    #             false_negatives += 1
    #
    #     print("Actual number of false negatives:", false_negatives)

    # def test_3nodes(self):
    #     gdag = cd.GaussDAG([1, 2, 3], arcs={(1, 2), (2, 3)})
    #     samples = gdag.sample_interventional({1: cd.ConstantIntervention(0)})
    #     estimated_perm, estimated_imap = igsp(samples, [3, 2, 1])


if __name__ == '__main__':
    unittest.main()
