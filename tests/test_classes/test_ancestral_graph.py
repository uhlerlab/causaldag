from unittest import TestCase
import unittest
import numpy as np
import subprocess
import os
import causaldag as cd
import random

CURR_DIR = os.path.dirname(__file__)


class TestAncestralGraph(TestCase):
    def setUp(self):
        self.d = cd.AncestralGraph(directed={(1, 3)}, bidirected={(3, 4)}, undirected={(1, 2)})

    def test_children(self):
        self.assertEqual(self.d.children_of(1), {3})
        self.assertEqual(self.d.children_of(2), set())
        self.assertEqual(self.d.children_of(3), set())
        self.assertEqual(self.d.children_of(4), set())

    def test_parents(self):
        self.assertEqual(self.d.parents_of(1), set())
        self.assertEqual(self.d.parents_of(2), set())
        self.assertEqual(self.d.parents_of(3), {1})
        self.assertEqual(self.d.parents_of(4), set())

    def test_spouses(self):
        self.assertEqual(self.d.spouses_of(1), set())
        self.assertEqual(self.d.spouses_of(2), set())
        self.assertEqual(self.d.spouses_of(3), {4})
        self.assertEqual(self.d.spouses_of(4), {3})

    def test_neighbors(self):
        self.assertEqual(self.d.neighbors_of(1), {2})
        self.assertEqual(self.d.neighbors_of(2), {1})
        self.assertEqual(self.d.neighbors_of(3), set())
        self.assertEqual(self.d.neighbors_of(4), set())

    def test_add_node(self):
        self.d.add_node(5)
        self.assertEqual(self.d.nodes, set(range(1, 6)))

    def test_add_directed(self):
        self.d.add_directed(1, 4)
        self.assertEqual(self.d.directed, {(1, 3), (1, 4)})

    # def test_add_adjacent_error(self):
    #     # === TEST TRYING TO OVERWRITE BIDIRECTED EDGE
    #     with self.assertRaises(cd.classes.ancestral_graph.AdjacentError) as cm:
    #         self.d.add_directed(3, 4)
    #     with self.assertRaises(cd.classes.ancestral_graph.AdjacentError) as cm:
    #         self.d.add_directed(4, 3)
    #     with self.assertRaises(cd.classes.ancestral_graph.AdjacentError) as cm:
    #         self.d.add_undirected(3, 4)
    #
    #     # === TEST TRYING TO OVERWRITE DIRECTED EDGE
    #     with self.assertRaises(cd.classes.ancestral_graph.AdjacentError) as cm:
    #         self.d.add_bidirected(1, 3)
    #     with self.assertRaises(cd.classes.ancestral_graph.AdjacentError) as cm:
    #         self.d.add_undirected(1, 3)
    #
    #     # === TEST TRYING TO OVERWRITE UNDIRECTED EDGE
    #     with self.assertRaises(cd.classes.ancestral_graph.AdjacentError) as cm:
    #         self.d.add_directed(1, 2)
    #     with self.assertRaises(cd.classes.ancestral_graph.AdjacentError) as cm:
    #         self.d.add_directed(2, 1)
    #     with self.assertRaises(cd.classes.ancestral_graph.AdjacentError) as cm:
    #         self.d.add_bidirected(2, 1)

    # def test_add_cycle_error(self):
    #     d = cd.AncestralGraph(directed={(1, 2), (2, 3)})
    #     with self.assertRaises(cd.classes.ancestral_graph.CycleError) as cm:
    #         d.add_directed(3, 1)
    #
    # def test_add_spouse_error(self):
    #     d = cd.AncestralGraph(directed={(1, 2), (2, 3)})
    #     with self.assertRaises(cd.classes.ancestral_graph.SpouseError) as cm:
    #         d.add_bidirected(3, 1)
    #
    # def test_add_neighbor_error(self):
    #     d = cd.AncestralGraph(directed={(1, 2), (2, 3)})
    #     with self.assertRaises(cd.classes.ancestral_graph.SpouseError) as cm:
    #         d.add_bidirected(3, 1)

    def test_msep_from_given(self):
        d = cd.AncestralGraph(directed={(1, 2), (3, 2), (2, 4), (3, 4)})
        # print(d.msep_from_given(1))
        # print(d.msep_from_given(1, 2))

    def test_msep(self):
        return
        # 1 -> 3 <-> 4 <- 2
        d = cd.AncestralGraph(directed={(1, 3), (2, 4)}, bidirected={(3, 4)})
        self.assertTrue(d.msep({1, 3}, 2))
        self.assertTrue(d.msep(1, 2))
        self.assertTrue(d.msep({1}, 4))
        self.assertFalse(d.msep({1, 3}, 4))
        self.assertFalse(d.msep(1, 2, {3, 4}))

        # undirected 4-cycle
        d = cd.AncestralGraph(undirected={(1, 2), (2, 3), (3, 4), (4, 1)})
        self.assertFalse(d.msep(1, 3))
        self.assertTrue(d.msep(1, 3, {2, 4}))

        # bidirected 4-cycle
        d = cd.AncestralGraph(bidirected={(1, 2), (2, 3), (3, 4), (4, 1)})
        self.assertTrue(d.msep(1, 3))
        self.assertFalse(d.msep(1, 3, 2))

        # discriminating path with discriminated node (3) as collider
        d = cd.AncestralGraph(directed={(1, 2), (2, 4)}, bidirected={(2, 3), (3, 4)})
        self.assertTrue(d.msep(1, 4, 2))
        self.assertFalse(d.msep(1, 4, {2, 3}))

        # big random graph
        np.random.seed(1729)
        random.seed(1729)
        nnodes = 10
        nodes = set(range(nnodes))
        g = cd.rand.directed_erdos(nnodes, 1/(nnodes-1))
        print(g.arcs)
        amat_file = os.path.join(CURR_DIR, '../data/random_mag.txt')
        np.savetxt(amat_file, g.to_amat(list(nodes))[0])

        rfile = os.path.join(CURR_DIR, '../R_scripts/test_msep.R')

        ntests = 50
        for _ in range(ntests):
            set_size = random.randint(1, 3)  # these currently need to be the same size b/c ggm is buggy
            nodes1 = random.sample(nodes, set_size)
            nodes2 = random.sample(nodes - set(nodes1), set_size)
            cond_set = random.sample(nodes - set(nodes1) - set(nodes2), random.randint(1, 3))
            print(nodes1, nodes2, cond_set)
            nodes1_str = ','.join(map(str, nodes1))
            nodes2_str = ','.join(map(str, nodes2))
            cond_set_str = ','.join(map(str, cond_set))
            if len(cond_set) > 0:
                r_output = subprocess.check_output(['Rscript', rfile, amat_file, nodes1_str, nodes2_str, cond_set_str])
            else:
                r_output = subprocess.check_output(['Rscript', rfile, amat_file, nodes1_str, nodes2_str])
            print(r_output.decode())
            r_output = r_output.decode() == 'TRUE'
            my_output = g.dsep(nodes1, nodes2, cond_set)
            print(r_output, my_output)
            self.assertEqual(r_output, my_output)

    # def dsep_regression_test1(self):
    #     d = cd.DAG(nodes=set(range(10)), arcs={(0, 9), (1, 9), (8, 9), (2, 5)})
    #     print(d.dsep({5, 9, 1}, {3, 0, 6}, {7, 2}, verbose=True))

    def test_disc_paths(self):
        g = cd.AncestralGraph(nodes=set(range(1, 5)), directed={(1, 2), (2, 4), (3, 2), (3, 4)})
        disc_paths = g.discriminating_paths()
        self.assertEqual(disc_paths, {(1, 2, 3, 4): 'n'})

        g = cd.AncestralGraph(nodes=set(range(1, 5)), directed={(1, 2), (2, 4)}, bidirected={(3, 2), (3, 4)})
        disc_paths = g.discriminating_paths()
        self.assertEqual(disc_paths, {(1, 2, 3, 4): 'c'})

        g = cd.AncestralGraph(nodes=set(range(1, 6)), directed={(1, 2), (2, 5), (3, 5)}, bidirected={(2, 3), (3, 4), (4, 5)})
        disc_paths = g.discriminating_paths()
        # print(disc_paths)
        self.assertEqual(disc_paths, {(1, 2, 3, 5): 'n', (1, 2, 3, 4, 5): 'c'})

    def test_legitimate_mark_changes(self):
        g = cd.AncestralGraph(directed={(0, 1)}, bidirected={(1, 2)})
        lmcs = g.legitimate_mark_changes()
        self.assertEqual(lmcs, ({(0, 1)}, {(2, 1)}))

        g = cd.AncestralGraph(directed={(0, 1), (1, 2)})
        lmcs = g.legitimate_mark_changes()
        self.assertEqual(lmcs, ({(0, 1)}, set()))

        g = cd.AncestralGraph(directed={(2, 1), (2, 3), (3, 5)}, bidirected={(1, 3), (4, 5), (2, 4)})
        lmcs = g.legitimate_mark_changes()
        self.assertEqual(lmcs, (set(), {(1, 3), (2, 4), (3, 1)}))

    def test_ancestor_dict(self):
        g = cd.AncestralGraph(bidirected={(0, 1)}, directed={(0, 2), (1, 3), (2, 4), (3, 4)})
        ancestor_dict = g.ancestor_dict()
        self.assertEqual(ancestor_dict[0], set())
        self.assertEqual(ancestor_dict[1], set())
        self.assertEqual(ancestor_dict[2], {0})
        self.assertEqual(ancestor_dict[3], {1})
        self.assertEqual(ancestor_dict[4], {0, 1, 2, 3})

    def test_fast_markov_equivalence_simple(self):
        g1 = cd.AncestralGraph(directed={(0, 1), (1, 3)}, bidirected={(1, 2), (2, 3)})
        g2 = cd.AncestralGraph(directed={(0, 1), (1, 3), (1, 2)}, bidirected={(2, 3)})
        g3 = cd.AncestralGraph(directed={(0, 1), (1, 2), (1, 3), (3, 2)})
        self.assertFalse(g1.fast_markov_equivalent(g2))
        self.assertFalse(g1.fast_markov_equivalent(g3))
        self.assertTrue(g2.fast_markov_equivalent(g3))

    def test_fast_markov_equivalence_all(self):
        s = 123231
        random.seed(s)
        np.random.seed(s)
        d = cd.rand.directed_erdos(10, exp_nbrs=2)
        g = d.marginal_mag({0, 1})
        for g_ in g.get_all_mec():
            self.assertTrue(g.fast_markov_equivalent(g_))


if __name__ == '__main__':
    unittest.main()
