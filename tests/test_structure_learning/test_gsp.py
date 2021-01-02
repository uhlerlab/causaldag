from unittest import TestCase
import unittest
import numpy as np
import causaldag as cd
import causaldag.classes.interventions
from causaldag.structure_learning import gsp, permutation2dag, unknown_target_igsp, igsp
from causaldag.utils.ci_tests import kci_test_vector, ki_test_vector, partial_correlation_test
from causaldag.utils.invariance_tests import kci_invariance_test, hsic_invariance_test
import random
from tqdm import tqdm
import random
np.random.seed(1729)
random.seed(1729)
from functools import partial
kci_no_regress = partial(kci_invariance_test, regress=False)


class TestGSP(TestCase):
    def test_gsp(self):
        ndags = 10
        nnodes = 8
        nsamples = 500
        nneighbors_list = list(range(1, 3))

        mean_shds = []
        percent_consistent = []
        for nneighbors in nneighbors_list:
            print('=== nneighbors = %s ===' % nneighbors)
            print("generating DAGs")
            dags = cd.rand.directed_erdos(nnodes, nneighbors/(nnodes-1), ndags)
            print("generating weights")
            gdags = [cd.rand.rand_weights(dag) for dag in dags]
            print("generating samples")
            samples_by_dag = [gdag.sample(nsamples) for gdag in gdags]
            print("computing correlation matrices")
            corr_by_dag = [np.corrcoef(samples, rowvar=False) for samples in samples_by_dag]
            print("running GSP")
            est_dags_and_summaries = [
                gsp(dict(C=corr, n=nsamples), nnodes, gauss_ci_test, depth=4, nruns=10, alpha=.01)
                for corr in corr_by_dag
            ]
            est_dags, summaries = zip(*est_dags_and_summaries)
            # print([str(d) for d in est_dags])
            shd_by_dag = np.array([est_dag.shd_skeleton(dag) for est_dag, dag in zip(est_dags, dags)])
            match_by_dag = np.sum(shd_by_dag == 0)
            mean_shds.append(shd_by_dag.mean())
            percent_consistent.append(match_by_dag/ndags)
        print("Mean SHDs:", mean_shds)
        print("Percent consistent:", percent_consistent)

    # def test_igsp_two_nodes(self):
    #     ndags = 50
    #     nsamples = 500
    #     dag = cd.DAG(arcs={(0, 1)})
    #     gdags = [cd.rand.rand_weights(dag) for i in range(ndags)]
    #     samples_list = [{
    #         frozenset(): gdag.sample(nsamples),
    #         frozenset({1}): gdag.sample_interventional_soft({1: cd.ScalingIntervention(.1)}, nsamples)
    #     } for gdag in gdags]
    #     corrs = [np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_list]
    #     est_dags = [
    #         igsp(samples, dict(C=corr, n=nsamples), 2, gauss_ci_test, kci_invariance_test)
    #         for samples, corr in zip(samples_list, corrs)
    #     ]
    #     is_mec = [est_dag.markov_equivalent(dag, {1}) for est_dag in est_dags]
    #     print(sum(is_mec))

    # def test_igsp_three_nodes(self):
    #     ndags = 10
    #     nsamples = 500
    #     dag = cd.DAG(arcs={(0, 1), (0, 2)})
    #     gdags = [cd.GaussDAG(nodes=[0, 1, 2], arcs=dag.arcs) for i in range(ndags)]
    #     # gdags = [cd.rand.rand_weights(dag) for i in range(ndags)]
    #     samples_list = [{
    #         frozenset(): gdag.sample(nsamples),
    #         frozenset({1}): gdag.sample_interventional_soft({1: cd.ScalingIntervention(.1)}, nsamples)
    #     } for gdag in gdags]
    #     corrs = [np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_list]
    #     est_dags = [
    #         igsp(samples, dict(C=corr, n=nsamples), 3, gauss_ci_test, kci_invariance_test)
    #         for samples, corr in zip(samples_list, corrs)
    #     ]
    #     print(est_dags)
    #     is_mec = [est_dag.markov_equivalent(dag, {1}) for est_dag in est_dags]
    #     print(sum(is_mec))

    # def test_igsp_line(self):
    #     ndags = 10
    #     nsamples = 200
    #     dag = cd.DAG(arcs={(0, 1), (1, 2), (2, 3), (3, 4)})
    #     gdags = [cd.GaussDAG(nodes=[0, 1, 2, 3, 4], arcs=dag.arcs) for i in range(ndags)]
    #     # gdags = [cd.rand.rand_weights(dag) for i in range(ndags)]
    #     samples_list = [{
    #         frozenset(): gdag.sample(nsamples),
    #         frozenset({0}): gdag.sample_interventional_soft({0: cd.ScalingIntervention(.1, .2)}, nsamples)
    #     } for gdag in gdags]
    #     corrs = [np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_list]
    #     est_dags = [
    #         igsp(samples, dict(C=corr, n=nsamples), 5, gauss_ci_test, hsic_invariance_test, nruns=10, alpha_invariance=1e-2)
    #         for samples, corr in zip(samples_list, corrs)
    #     ]
    #     shds = [dag.shd(est_dag) for est_dag in est_dags]
    #     print(est_dags)
    #     print(shds)
    #     print(np.mean(shds))

    # def test_utigsp_line(self):
    #     ndags = 30
    #     nsamples = 500
    #     dag = cd.DAG(arcs={(0, 1), (1, 2), (2, 3), (3, 4)})
    #     gdags = [cd.GaussDAG(nodes=list(range(5)), arcs=dag.arcs) for i in range(ndags)]
    #     # gdags = [cd.rand.rand_weights(dag) for i in range(ndags)]
    #     samples_list = [{
    #         frozenset(): gdag.sample(nsamples),
    #         frozenset({4}): gdag.sample_interventional_soft({0: cd.ScalingIntervention(.1, .01), 4: cd.ScalingIntervention(.1, .01)}, nsamples)
    #     } for gdag in gdags]
    #     corrs = [np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_list]
    #     est_dags = [
    #         unknown_target_igsp(samples, dict(C=corr, n=nsamples), 5, gauss_ci_test, hsic_invariance_test, nruns=10, alpha_invariance=1e-5)
    #         for samples, corr in zip(samples_list, corrs)
    #     ]
    #     shds = [dag.shd(est_dag) for est_dag in est_dags]
    #     print(est_dags)
    #     print(shds)
    #     print(np.mean(shds))

    # def test_igsp_three_nodes_complete(self):
    #     ndags = 30
    #     nsamples = 100
    #     gdag = cd.GaussDAG(nodes=[0, 1, 2], arcs={(0, 1): 2, (0, 2): 3, (1, 2): 5})
    #     samples_list = [{
    #         frozenset(): gdag.sample(nsamples),
    #         frozenset({1}): gdag.sample_interventional_perfect({1: cd.GaussIntervention(0, 1)}, nsamples)
    #     } for i in range(ndags)]
    #     corrs = [np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_list]
    #     ALPHA = 1e-5
    #     est_dags = [
    #         igsp(samples, dict(C=corr, n=nsamples), 3, gauss_ci_test, hsic_invariance_test, nruns=10, alpha=1e-5, alpha_invariance=ALPHA)
    #         for samples, corr in zip(samples_list, corrs)
    #     ]
    #     print(est_dags)
    #     is_mec = [est_dag.markov_equivalent(gdag.to_dag(), {1}) for est_dag in est_dags]
    #     print(sum(is_mec))

    # def test_igsp_ten_nodes(self):
    #     ndags = 50
    #     nsamples = 500
    #     nnodes = 20
    #     nsettings_list = [nnodes]
    #     dags = cd.rand.directed_erdos(nnodes, 1.5/(nnodes-1), ndags)
    #     gdags = [cd.rand.rand_weights(dag) for dag in dags]
    #
    #     avg_shds_skel = []
    #     avg_shds = []
    #     percent_meq = []
    #     percent_imeq = []
    #     percent_consistent = []
    #     for nsettings in nsettings_list:
    #         print('====')
    #         iv_nodes_list = [random.sample(list(range(nnodes)), nsettings) for _ in gdags]
    #         samples_list = []
    #         for gdag, iv_nodes in zip(gdags, iv_nodes_list):
    #             samples = {frozenset(): gdag.sample(nsamples)}
    #             for iv_node in iv_nodes:
    #                 samples[frozenset({iv_node})] = gdag.sample_interventional_perfect({iv_node: causaldag.classes.interventions.ConstantIntervention(5)}, nsamples)
    #             samples_list.append(samples)
    #         corrs = [np.corrcoef(samples[frozenset()], rowvar=False) for samples in samples_list]
    #         est_dags = []
    #         for dag, gdag, samples, corr in tqdm(zip(dags, gdags, samples_list, corrs), total=ndags):
    #             est_dag = igsp(samples, dict(C=corr, n=nsamples), nnodes, gauss_ci_test, kci_no_regress, nruns=10, alpha=1e-5, alpha_invariance=1e-5)
    #             est_dags.append(est_dag)
    #             print(dag.shd(est_dag))
    #         shds_skel = [est_dag.shd_skeleton(dag) for est_dag, dag in zip(est_dags, dags)]
    #         shds = [est_dag.shd(dag) for est_dag, dag in zip(est_dags, dags)]
    #         markov_equiv = [est_dag.markov_equivalent(dag) for est_dag, dag in zip(est_dags, dags)]
    #         imarkov_equiv = [est_dag.markov_equivalent(dag, iv_nodes) for est_dag, dag, iv_nodes in zip(est_dags, dags, iv_nodes_list)]
    #         consistent = [est_dag == dag for est_dag, dag in zip(est_dags, dags)]
    #
    #         avg_shds_skel.append(np.mean(shds_skel))
    #         avg_shds.append(np.mean(shds))
    #         percent_meq.append(np.mean(markov_equiv))
    #         percent_imeq.append(np.mean(imarkov_equiv))
    #         percent_consistent.append(np.mean(consistent))
    #
    #     print("Average SHD", avg_shds)
    #     print("Average SHD skel", avg_shds_skel)
    #     print("Percent correct MEC", percent_meq)
    #     print("Percent correct I-MEC", percent_imeq)
    #     print("Percent consistent", percent_consistent)

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
