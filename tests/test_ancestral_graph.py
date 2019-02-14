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

    def test_add_cycle_error(self):
        d = cd.AncestralGraph(directed={(1, 2), (2, 3)})
        with self.assertRaises(cd.classes.ancestral_graph.CycleError) as cm:
            d.add_directed(3, 1)

    def test_add_spouse_error(self):
        d = cd.AncestralGraph(directed={(1, 2), (2, 3)})
        with self.assertRaises(cd.classes.ancestral_graph.SpouseError) as cm:
            d.add_bidirected(3, 1)

    def test_add_neighbor_error(self):
        d = cd.AncestralGraph(directed={(1, 2), (2, 3)})
        with self.assertRaises(cd.classes.ancestral_graph.SpouseError) as cm:
            d.add_bidirected(3, 1)


if __name__ == '__main__':
    unittest.main()
