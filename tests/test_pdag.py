# Author: Chandler Squires
from unittest import TestCase
import unittest
import os
import causaldag as cd
import numpy as np
import itertools as itr
import random


class TestDAG(TestCase):
    def test_copy(self):
        pdag = cd.PDAG(arcs={(1, 2), (1, 3), (2, 3)})
        pdag2 = pdag.copy()
        self.assertEqual(pdag.arcs, pdag2.arcs)
        self.assertEqual(pdag.edges, pdag2.edges)
        self.assertEqual(pdag.neighbors, pdag2.neighbors)
        self.assertEqual(pdag.parents, pdag2.parents)
        self.assertEqual(pdag.children, pdag2.children)

        pdag2._replace_arc_with_edge((1, 2))
        self.assertNotEqual(pdag.arcs, pdag2.arcs)
        self.assertNotEqual(pdag.edges, pdag2.edges)
        self.assertNotEqual(pdag.parents, pdag2.parents)
        self.assertNotEqual(pdag.children, pdag2.children)

    def test_cpdag_confounding(self):
        dag = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        cpdag = dag.cpdag()
        self.assertEqual(cpdag.arcs, set())
        self.assertEqual(cpdag.edges, {(1, 2), (1, 3), (2, 3)})
        self.assertEqual(cpdag.parents[1], set())
        self.assertEqual(cpdag.parents[2], set())
        self.assertEqual(cpdag.parents[3], set())
        self.assertEqual(cpdag.children[1], set())
        self.assertEqual(cpdag.children[2], set())
        self.assertEqual(cpdag.children[3], set())
        self.assertEqual(cpdag.neighbors[1], {2, 3})
        self.assertEqual(cpdag.neighbors[2], {1, 3})
        self.assertEqual(cpdag.neighbors[3], {1, 2})
        self.assertEqual(cpdag.undirected_neighbors[1], {2, 3})
        self.assertEqual(cpdag.undirected_neighbors[2], {1, 3})
        self.assertEqual(cpdag.undirected_neighbors[3], {1, 2})

        self.assertEqual(dag.arcs, {(1, 2), (1, 3), (2, 3)})
        self.assertEqual(dag.parents[1], set())
        self.assertEqual(dag.parents[2], {1})
        self.assertEqual(dag.parents[3], {1, 2})
        self.assertEqual(dag.children[1], {2, 3})
        self.assertEqual(dag.children[2], {3})
        self.assertEqual(dag.children[3], set())

    def test_cpdag_v(self):
        dag = cd.DAG(arcs={(1, 2), (3, 2)})
        cpdag = dag.cpdag()
        self.assertEqual(cpdag.arcs, {(1, 2), (3, 2)})
        self.assertEqual(cpdag.edges, set())
        self.assertEqual(cpdag.parents[1], set())
        self.assertEqual(cpdag.parents[2], {1, 3})
        self.assertEqual(cpdag.parents[3], set())
        self.assertEqual(cpdag.children[1], {2})
        self.assertEqual(cpdag.children[2], set())
        self.assertEqual(cpdag.children[3], {2})
        self.assertEqual(cpdag.neighbors[1], {2})
        self.assertEqual(cpdag.neighbors[2], {1, 3})
        self.assertEqual(cpdag.neighbors[3], {2})
        self.assertEqual(cpdag.undirected_neighbors[1], set())
        self.assertEqual(cpdag.undirected_neighbors[2], set())
        self.assertEqual(cpdag.undirected_neighbors[3], set())

        self.assertEqual(dag.arcs, {(1, 2), (3, 2)})
        self.assertEqual(dag.parents[1], set())
        self.assertEqual(dag.parents[2], {1, 3})
        self.assertEqual(dag.parents[3], set())
        self.assertEqual(dag.children[1], {2})
        self.assertEqual(dag.children[2], set())
        self.assertEqual(dag.children[3], {2})

    # def test_cpdag_file(self):
    #     curr_folder = os.path.dirname(__file__)
    #     random.seed(1729)
    #     np.random.seed(1729)
    #     dag = cd.rand.directed_erdos(30, .7, 1)
    #     np.savetxt(os.path.join(curr_folder, './dag1.txt'), dag.to_amat())
    #     os.system('Rscript %s/test_dag2cpdag.R' % curr_folder)
    #     cpdag = dag.cpdag()
    #
    #     true_cpdag_edges = set()
    #     true_cpdag_arcs = set()
    #     true_cpdag_amat = np.loadtxt(os.path.join(curr_folder, './cpdag1.txt'))
    #     for i, j in zip(*np.triu_indices_from(true_cpdag_amat)):
    #         if true_cpdag_amat[i, j] == 1:
    #             if true_cpdag_amat[j, i] == 1:
    #                 true_cpdag_edges.add((i, j))
    #             else:
    #                 true_cpdag_arcs.add((i, j))
    #
    #     print(len(cpdag.edges))
    #     self.assertEqual(cpdag.arcs, true_cpdag_arcs)
    #     self.assertEqual(cpdag.edges, true_cpdag_edges)

    # def test_icpdag_file(self):
    #     curr_folder = os.path.dirname(__file__)
    #     random.seed(1729)
    #     np.random.seed(1729)
    #     dag = cd.rand.directed_erdos(30, .7, 1)
    #     interventions = [random.sample(list(range(100)), 5) for _ in range(10)]
    #     np.savetxt(os.path.join(curr_folder, './dag1.txt'), dag.to_amat())
    #     np.savetxt(os.path.join(curr_folder, './interventions.txt'), interventions)
    #     os.system('Rscript %s/test_dag2icpdag.R' % curr_folder)
    #     icpdag = dag.interventional_cpdag(interventions, cpdag=dag.cpdag())
    #
    #     true_icpdag_edges = set()
    #     true_icpdag_arcs = set()
    #     true_icpdag_amat = np.loadtxt(os.path.join(curr_folder, './icpdag1.txt'))
    #     for i, j in zip(*np.triu_indices_from(true_icpdag_amat)):
    #         if true_icpdag_amat[i, j] == 1:
    #             if true_icpdag_amat[j, i] == 1:
    #                 true_icpdag_edges.add((i, j))
    #             else:
    #                 true_icpdag_arcs.add((i, j))
    #
    #     print('icpdag', len(icpdag.edges))
    #     self.assertEqual(icpdag.arcs, true_icpdag_arcs)
    #     self.assertEqual(icpdag.edges, true_icpdag_edges)

    def test_interventional_cpdag(self):
        dag = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        cpdag = dag.cpdag()
        int_cpdag = dag.interventional_cpdag([{1}], cpdag=cpdag)
        self.assertEqual(int_cpdag.arcs, {(1, 2), (1, 3)})
        self.assertEqual(int_cpdag.edges, {(2, 3)})
        self.assertEqual(int_cpdag.undirected_neighbors[1], set())
        self.assertEqual(int_cpdag.undirected_neighbors[2], {3})
        self.assertEqual(int_cpdag.undirected_neighbors[3], {2})

    # def test_add_known_arcs(self):
    #     dag = cd.DAG(arcs={(1, 2), (2, 3)})
    #     cpdag = dag.cpdag()
    #     self.assertEqual(cpdag.arcs, set())
    #     self.assertEqual(cpdag.edges, {(1, 2), (2, 3)})
    #     cpdag.add_known_arc(1, 2)
    #     self.assertEqual(cpdag.arcs, {(1, 2), (2, 3)})
    #     self.assertEqual(cpdag.edges, set())

    def test_pdag2alldags_3nodes_complete(self):
        dag = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        cpdag = dag.cpdag()
        dags = cpdag.all_dags(verbose=False)
        self.assertEqual(len(dags), 6)
        for dag in dags:
            self.assertEqual(len(dag), 3)
        true_possible_arcs = {
            frozenset({(1, 2), (1, 3), (2, 3)}),
            frozenset({(1, 2), (1, 3), (3, 2)}),  # flip 2->3
            frozenset({(1, 2), (3, 1), (3, 2)}),  # flip 1->3
            frozenset({(2, 1), (3, 1), (3, 2)}),  # flip 1->2
            frozenset({(2, 1), (3, 1), (2, 3)}),  # flip 3->2
            frozenset({(2, 1), (1, 3), (2, 3)}),  # flip 3->1
        }
        self.assertEqual(true_possible_arcs, dags)

    def test_pdag2alldags_3nodes_chain(self):
        dag = cd.DAG(arcs={(1, 2), (2, 3)})
        cpdag = dag.cpdag()
        dags = cpdag.all_dags(verbose=False)
        true_possible_arcs = {
            frozenset({(1, 2), (2, 3)}),
            frozenset({(2, 1), (2, 3)}),
            frozenset({(2, 1), (3, 2)}),
        }
        self.assertEqual(true_possible_arcs, dags)

    def test_pdag2alldags_5nodes(self):
        dag = cd.DAG(arcs={(1, 2), (2, 3), (1, 3), (2, 4), (2, 5), (3, 5), (4, 5)})
        cpdag = dag.cpdag()
        dags = cpdag.all_dags()
        for arcs in dags:
            dag2 = cd.DAG(arcs=set(arcs))
            cpdag2 = dag2.cpdag()
            if cpdag2 != cpdag:
                print(cpdag2.nodes, cpdag.nodes)
                print(cpdag2.arcs, cpdag.arcs)
                print(cpdag2.edges, cpdag.edges)
            self.assertEqual(cpdag, cpdag2)

    def test_pdag2alldags_6nodes_complete(self):
        dag = cd.DAG(arcs={(i, j) for i, j in itr.combinations(range(6), 2)})
        cpdag = dag.cpdag()
        dags = cpdag.all_dags()
        self.assertEqual(len(dags), np.prod(range(1, 7)))

    def test_icpdag2alldags(self):
        dag = cd.DAG(arcs={(1, 2), (1, 3), (2, 3), (4, 5)})
        icpdag = dag.interventional_cpdag([{2}], cpdag=dag.cpdag())
        dags = icpdag.all_dags()
        self.assertEqual(len(dags), 2)

    def test_optimal_intervention_1intervention(self):
        dag = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        best_ivs, icpdags = dag.optimal_intervention_greedy()
        self.assertEqual(best_ivs, [2])
        self.assertEqual(icpdags[0].arcs, dag.arcs)

    def test_optimal_intervention_2interventions(self):
        dag = cd.DAG(arcs={(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)})
        best_ivs, icpdags = dag.optimal_intervention_greedy(num_interventions=2)
        self.assertEqual(best_ivs, [2, None])
        self.assertEqual(icpdags[0].arcs, dag.arcs)
        self.assertEqual(icpdags[1].arcs, dag.arcs)

    def test_optimal_intervention_2interventions2(self):
        dag = cd.DAG(arcs={(1, 2), (1, 3), (2, 3), (4, 5)})
        best_ivs, icpdags = dag.optimal_intervention_greedy(num_interventions=2)
        self.assertEqual(best_ivs, [2, 4])
        self.assertEqual(icpdags[0].arcs, {(1, 2), (1, 3), (2, 3)})
        self.assertEqual(icpdags[1].arcs, {(1, 2), (1, 3), (2, 3), (4, 5)})

    def test_fully_orienting_interventions_6nodes_complete(self):
        dag = cd.DAG(arcs={(i, j) for i, j in itr.combinations(range(6), 2)})
        ivs, icpdags = dag.fully_orienting_interventions_greedy()

    def test_to_dag(self):
        dag = cd.DAG(arcs={(1, 2), (2, 3)})
        cpdag = dag.cpdag()
        dag2 = cpdag.to_dag()
        true_possible_arcs = {
            frozenset({(1, 2), (2, 3)}),
            frozenset({(2, 1), (2, 3)}),
            frozenset({(2, 1), (3, 2)}),
        }
        self.assertIn(frozenset(dag2.arcs), true_possible_arcs)

    def test_to_dag_complete3(self):
        dag = cd.DAG(arcs={(1, 2), (2, 3), (1, 3)})
        cpdag = dag.cpdag()
        dag2 = cpdag.to_dag()
        true_possible_arcs = {
            frozenset({(1, 2), (1, 3), (2, 3)}),
            frozenset({(1, 2), (1, 3), (3, 2)}),  # flip 2->3
            frozenset({(1, 2), (3, 1), (3, 2)}),  # flip 1->3
            frozenset({(2, 1), (3, 1), (3, 2)}),  # flip 1->2
            frozenset({(2, 1), (3, 1), (2, 3)}),  # flip 3->2
            frozenset({(2, 1), (1, 3), (2, 3)}),  # flip 3->1
        }
        self.assertIn(frozenset(dag2.arcs), true_possible_arcs)

    # def test_interventional_cpdag_no_obs(self):
    #     dag = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
    #
    #     icpdag1 = dag.interventional_cpdag([2])
    #     self.assertEqual(icpdag1.arcs, {(1, 3), (2, 3)})
    #     self.assertEqual(icpdag1.edges, set())
    #
    #     icpdag2 = dag.interventional_cpdag([1])
    #     self.assertEqual(icpdag2.arcs, set())
    #     self.assertEqual(icpdag2.edges, {(1, 2), (1, 3), (2, 3)})

    def test_from_amat(self):
        amat = np.zeros([3, 3])
        amat[0, 1] = 1
        amat[0, 2] = 1
        amat[1, 2] = 1
        amat[2, 1] = 1
        pdag = cd.PDAG.from_amat(amat)
        self.assertEqual(pdag.arcs, {(0, 1), (0, 2)})
        self.assertEqual(pdag.edges, {(1, 2)})

    def test_shd(self):
        d1 = cd.PDAG(arcs={(0, 1)})
        d2 = cd.PDAG()
        self.assertEqual(d1.shd(d2), 1)
        self.assertEqual(d2.shd(d1), 1)

        d1 = cd.PDAG(arcs={(0, 1)})
        d2 = cd.PDAG(arcs={(0, 1)})
        self.assertEqual(d1.shd(d2), 0)
        self.assertEqual(d2.shd(d1), 0)

        d1 = cd.PDAG(arcs={(0, 1)})
        d2 = cd.PDAG(edges={(0, 1)})
        self.assertEqual(d1.shd(d2), 1)
        self.assertEqual(d2.shd(d1), 1)

        d1 = cd.PDAG(arcs={(0, 1)})
        d2 = cd.PDAG(arcs={(1, 0)})
        self.assertEqual(d1.shd(d2), 1)
        self.assertEqual(d2.shd(d1), 1)

        d1 = cd.PDAG(arcs={(0, 1), (0, 2)})
        d2 = cd.PDAG(arcs={(0, 1)}, edges={(0, 2)})
        self.assertEqual(d1.shd(d2), 1)
        self.assertEqual(d2.shd(d1), 1)

        d1 = cd.PDAG(arcs={(0, 1), (0, 2)})
        d2 = cd.PDAG(arcs={(1, 0), (2, 0)})
        self.assertEqual(d1.shd(d2), 2)
        self.assertEqual(d2.shd(d1), 2)

        d1 = cd.PDAG(arcs={(0, 1), (0, 2)})
        d2 = cd.PDAG(edges={(1, 0), (2, 0)})
        self.assertEqual(d1.shd(d2), 2)
        self.assertEqual(d2.shd(d1), 2)

    def test_to_complete_pdag(self):
        g = cd.PDAG(arcs={(0, 1), (0, 2), (3, 0)}, edges={(1, 2), (1, 3), (2, 3)})
        g.to_complete_pdag()
        g_comp = cd.PDAG(arcs={(0, 1), (0, 2), (3, 0), (3, 1), (3, 2)}, edges={(1, 2)})
        self.assertEqual(g, g_comp)

        g = cd.PDAG(arcs={(0, 1), (0, 2), (0, 3)}, edges={(1, 2), (1, 3), (2, 3)})
        g.to_complete_pdag()
        g_comp = cd.PDAG(arcs={(0, 1), (0, 2), (0, 3)}, edges={(1, 2), (1, 3), (2, 3)})
        self.assertEqual(g, g_comp)

        g = cd.PDAG(arcs={(1, 0), (2, 0), (3, 0)}, edges={(1, 2), (1, 3), (2, 3)})
        g.to_complete_pdag()
        g_comp = cd.PDAG(arcs={(1, 0), (2, 0), (3, 0)}, edges={(1, 2), (1, 3), (2, 3)})
        self.assertEqual(g, g_comp)


if __name__ == '__main__':
    unittest.main()


