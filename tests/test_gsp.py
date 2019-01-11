from unittest import TestCase
import unittest
import numpy as np
import causaldag as cd
from causaldag.inference.structural import gsp, perm2dag, is_icovered, unknown_target_igsp, igsp
from causaldag.utils.ci_tests import kci_test_vector, ki_test_vector, gauss_ci_test, kci_invariance_test
import random
from tqdm import tqdm
import random
np.random.seed(1729)
random.seed(1729)


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

    def test_igsp_two_nodes(self):
        ndags = 50
        nsamples = 1000
        dag = cd.DAG(arcs={(0, 1)})
        gdags = [cd.rand.rand_weights(dag) for i in range(ndags)]
        samples_list = [{
            frozenset(): gdag.sample(nsamples),
            frozenset({1}): gdag.sample_interventional({frozenset({1}): cd.GaussIntervention(0, 2)}, nsamples)
        } for gdag in gdags]
        corrs = [np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_list]
        est_dags = [
            igsp(samples, dict(C=corr, n=nsamples), 2, gauss_ci_test, kci_invariance_test)
            for samples, corr in zip(samples_list, corrs)
        ]

        for est_dag in est_dags:
            print(est_dag.markov_equivalent(dag, {1}))

    def test_igsp_ten_nodes(self):
        ndags = 50
        nsamples = 500
        nnodes = 10
        nsettings_list = [1, 2, 4, 6, 8, 10]
        dags = cd.rand.directed_erdos(nnodes, 1.5/9, ndags)
        gdags = [cd.rand.rand_weights(dag) for dag in dags]

        avg_shds_skel = []
        avg_shds = []
        for nsettings in nsettings_list:
            samples_list = []
            iv_nodes_list = [random.sample(list(range(nnodes)), nsettings) for _ in gdags]
            for gdag, iv_nodes in zip(gdags, iv_nodes_list):
                samples = {frozenset(): gdag.sample(nsamples)}
                for iv_node in iv_nodes:
                    samples[frozenset({iv_node})] = gdag.sample_interventional({frozenset({iv_node}): cd.ConstantIntervention(0)}, nsamples)
                samples_list.append(samples)
            corrs = [np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_list]
            est_dags = [
                igsp(samples, dict(C=corr, n=nsamples), nnodes, gauss_ci_test, kci_invariance_test)
                for samples, corr in zip(samples_list, corrs)
            ]
            shds_skel = [est_dag.shd_skeleton(dag) for est_dag, dag in zip(est_dags, dags)]
            shds = [est_dag.shd(dag) for est_dag, dag in zip(est_dags, dags)]
            markov_equiv = [est_dag.markov_equivalent(dag) for est_dag, dag in zip(est_dags, dags)]
            print(sum(markov_equiv))
            imarkov_equiv = [est_dag.markov_equivalent(dag, iv_nodes) for est_dag, dag, iv_nodes in zip(est_dags, dags, iv_nodes_list)]
            print(sum(imarkov_equiv))
            avg_shds_skel.append(np.mean(shds_skel))
            avg_shds.append(np.mean(shds))
        print(avg_shds)
        print(avg_shds_skel)

    # def test_igsp(self):
    #     ndags = 50
    #     nnodes = 10
    #     nsamples = 1000
    #     nneighbors_list = [1.5]
    #
    #     mean_shds = []
    #     percent_consistent = []
    #     for nneighbors in nneighbors_list:
    #         print('=== nneighbors = %s ===' % nneighbors)
    #         print("generating DAGs")
    #         dags = cd.rand.directed_erdos(nnodes, nneighbors/(nnodes-1), ndags)
    #         # print([str(dag) for dag in dags])
    #         print("generating weights")
    #         gdags = [cd.rand.rand_weights(dag) for dag in dags]
    #         print("choosing interventions")
    #         # dag2iv_nodes = [frozenset({random.randint(0, nnodes)}) for i in range(ndags)]
    #         dag2iv_nodes = [frozenset({1}) for i in range(ndags)]
    #         print("calculating icpdags")
    #         cpdags = [dag.cpdag() for dag in dags]
    #         icpdags = [dag.interventional_cpdag(iv_nodes, cpdag=cpdag) for dag, iv_nodes, cpdag in zip(dags, dag2iv_nodes, cpdags)]
    #         print("generating samples")
    #         samples_by_dag = [{
    #             frozenset(): gdag.sample(nsamples),
    #             iv_nodes: gdag.sample_interventional({iv_nodes: cd.GaussIntervention(0, 2)}, nsamples=nsamples)
    #         } for gdag, iv_nodes in zip(gdags, dag2iv_nodes)]
    #         print("computing correlation matrices")
    #         corr_by_dag = [np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_by_dag]
    #         print("running GSP")
    #         est_dags = [
    #             igsp(
    #                 samples,
    #                 dict(C=corr, n=nsamples),
    #                 nnodes,
    #                 gauss_ci_test,
    #                 kci_invariance_test,
    #                 depth=4,
    #                 nruns=10,
    #                 alpha=0.01,
    #                 alpha_invariance=0.05,
    #                 verbose=True
    #             )
    #             for samples, corr in zip(samples_by_dag, corr_by_dag)
    #         ]
    #         est_cpdags = [est_dag.cpdag() for est_dag in est_dags]
    #         est_icpdags = [est_dag.interventional_cpdag(iv_nodes, cpdag=cpdag) for est_dag, iv_nodes, cpdag in zip(est_dags, dag2iv_nodes, est_cpdags)]
    #         match_icpdag = [est_icpdag == icpdag for est_icpdag, icpdag in zip(est_icpdags, icpdags)]
    #         print(sum(match_icpdag))
    #         for dag, est_dag, icpdag, est_icpdag, match_icpdag in zip(dags, est_dags, icpdags, est_icpdags, match_icpdag):
    #             if not match_icpdag:
    #                 print(str(dag))
    #                 print(str(est_dag))
    #                 print(icpdag.arcs, icpdag.edges, icpdag.nodes)
    #                 print(est_icpdag.arcs, est_icpdag.edges, est_icpdag.nodes)
    #         # print([str(d) for d in est_dags])
    #         shd_by_dag = np.array([est_dag.shd_skeleton(dag) for est_dag, dag in zip(est_dags, dags)])
    #         match_by_dag = np.sum(shd_by_dag == 0)
    #         mean_shds.append(shd_by_dag.mean())
    #         percent_consistent.append(match_by_dag/ndags)
    #     print("Mean SHDs:", mean_shds)
    #     print("Percent consistent:", percent_consistent)

    # def test_utigsp(self):
    #     ndags = 50
    #     nnodes = 8
    #     nsamples = 300
    #     nneighbors_list = list(range(1, 2))
    #
    #     mean_shds = []
    #     percent_consistent = []
    #     for nneighbors in nneighbors_list:
    #         print('=== nneighbors = %s ===' % nneighbors)
    #         print("generating DAGs")
    #         dags = cd.rand.directed_erdos(nnodes, nneighbors / (nnodes - 1), ndags)
    #         print("generating weights")
    #         gdags = [cd.rand.rand_weights(dag) for dag in dags]
    #         print("generating samples")
    #         samples_by_dag = []
    #         for gdag in gdags:
    #             iv_nodes = frozenset({random.randint(0, nnodes)})
    #             samples_by_dag.append({
    #                 frozenset(): gdag.sample(nsamples),
    #                 iv_nodes: gdag.sample_interventional({iv_nodes: cd.GaussIntervention(0, 2)}, nsamples)
    #             })
    #         print("computing correlation matrices")
    #         corr_by_dag = list(tqdm((np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_by_dag), total=ndags))
    #         print("running UTIGSP")
    #         est_dags = list(tqdm((
    #             unknown_target_igsp(
    #                 samples,
    #                 dict(C=corr, n=nsamples),
    #                 nnodes,
    #                 gauss_ci_test,
    #                 kci_invariance_test,
    #                 depth=4,
    #                 nruns=10,
    #                 alpha=.01,
    #                 alpha_invariance=.05
    #             )
    #             for samples, corr in zip(samples_by_dag, corr_by_dag)
    #         ), total=ndags))
    #         # print([str(d) for d in est_dags])
    #         shd_by_dag = np.array([est_dag.shd_skeleton(dag) for est_dag, dag in zip(est_dags, dags)])
    #         match_by_dag = np.sum(shd_by_dag == 0)
    #         mean_shds.append(shd_by_dag.mean())
    #         percent_consistent.append(match_by_dag / ndags)
    #     print("Mean SHDs:", mean_shds)
    #     print("Percent consistent:", percent_consistent)

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
