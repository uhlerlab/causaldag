from unittest import TestCase
import unittest
import numpy as np
import causaldag as cd


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
        print(d.msep_from_given(1))
        print(d.msep_from_given(1, 2))

    def test_disc_paths(self):
        g = cd.AncestralGraph(nodes=set(range(1, 5)), directed={(1, 2), (2, 4), (3, 2), (3, 4)})
        disc_paths = g.discriminating_paths()
        self.assertEqual(disc_paths, [([1, 2, 3, 4], 'n')])

        g = cd.AncestralGraph(nodes=set(range(1, 5)), directed={(1, 2), (2, 4)}, bidirected={(3, 2), (3, 4)})
        disc_paths = g.discriminating_paths()
        self.assertEqual(disc_paths, [([1, 2, 3, 4], 'c')])

        g = cd.AncestralGraph(nodes=set(range(1, 6)), directed={(1, 2), (2, 5), (3, 5)}, bidirected={(2, 3), (3, 4), (4, 5)})
        disc_paths = g.discriminating_paths()
        print(disc_paths)
        self.assertEqual(disc_paths, [([1, 2, 3, 5], 'n'), ([1, 2, 3, 4, 5], 'c')])


if __name__ == '__main__':
    unittest.main()
