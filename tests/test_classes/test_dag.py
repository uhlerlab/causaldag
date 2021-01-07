from unittest import TestCase
import unittest
import numpy as np
import causaldag as cd


class TestDAG(TestCase):
    def setUp(self):
        self.d = cd.DAG(arcs={(1, 2), (1, 3), (3, 4), (2, 4), (3, 5)})

    def test_neighbors(self):
        self.assertEqual(self.d._neighbors[1], {2, 3})
        self.assertEqual(self.d._neighbors[2], {1, 4})
        self.assertEqual(self.d._neighbors[3], {1, 4, 5})
        self.assertEqual(self.d._neighbors[4], {2, 3})
        self.assertEqual(self.d._neighbors[5], {3})

    def test_children(self):
        self.assertEqual(self.d._children[1], {2, 3})
        self.assertEqual(self.d._children[2], {4})
        self.assertEqual(self.d._children[3], {4, 5})
        self.assertEqual(self.d._children[4], set())
        self.assertEqual(self.d._children[5], set())

    def test_parents(self):
        self.assertEqual(self.d._parents[1], set())
        self.assertEqual(self.d._parents[2], {1})
        self.assertEqual(self.d._parents[3], {1})
        self.assertEqual(self.d._parents[4], {2, 3})
        self.assertEqual(self.d._parents[5], {3})

    def test_downstream(self):
        self.assertEqual(self.d.descendants_of(1), {2, 3, 4, 5})
        self.assertEqual(self.d.descendants_of(2), {4})
        self.assertEqual(self.d.descendants_of(3), {4, 5})
        self.assertEqual(self.d.descendants_of(4), set())
        self.assertEqual(self.d.descendants_of(5), set())

    def test_upstream(self):
        self.assertEqual(self.d.ancestors_of(1), set())
        self.assertEqual(self.d.ancestors_of(2), {1})
        self.assertEqual(self.d.ancestors_of(3), {1})
        self.assertEqual(self.d.ancestors_of(4), {1, 2, 3})
        self.assertEqual(self.d.ancestors_of(5), {1, 3})

    def test_sources(self):
        self.assertEqual(self.d.sources(), {1})

    def test_sinks(self):
        self.assertEqual(self.d.sinks(), {4, 5})

    def test_add_node(self):
        self.d.add_node(6)
        self.assertEqual(self.d.nodes, set(range(1, 7)))

    def test_add_arc(self):
        self.d.add_arc(2, 3)
        self.assertEqual(self.d._children[2], {3, 4})
        self.assertEqual(self.d._neighbors[2], {1, 3, 4})
        self.assertEqual(self.d._parents[3], {1, 2})
        self.assertEqual(self.d._neighbors[3], {1, 2, 4, 5})
        self.assertEqual(self.d.descendants_of(2), {3, 4, 5})
        self.assertEqual(self.d.ancestors_of(3), {1, 2})

    def test_topological_sort(self):
        t = self.d.topological_sort()
        ixs = {node: t.index(node) for node in self.d.nodes}
        for i, j in self.d.arcs:
            self.assertTrue(ixs[i] < ixs[j])

    def test_add_arc_cycle(self):
        with self.assertRaises(cd.CycleError) as cm:
            self.d.add_arc(2, 1)
        self.assertEqual(cm.exception.cycle, [1, 2, 1])
        with self.assertRaises(cd.CycleError):
            self.d.add_arc(4, 1)
        with self.assertRaises(cd.CycleError) as cm:
            self.d.add_arc(5, 1)
        self.assertEqual(cm.exception.cycle, [1, 3, 5, 1])

    def test_interventional_cpdag_2node(self):
        d = cd.DAG(arcs={(0, 1)})
        c = d.interventional_cpdag([{1}], cpdag=d.cpdag())
        self.assertEqual(c.arcs, {(0, 1)})
        self.assertEqual(c.edges, set())
        c = d.interventional_cpdag([{0}], cpdag=d.cpdag())
        self.assertEqual(c.arcs, {(0, 1)})
        self.assertEqual(c.edges, set())

    def test_interventional_cpdag_3node(self):
        d = cd.DAG(arcs={(0, 1), (0, 2), (1, 2)})
        c = d.interventional_cpdag([{0}], cpdag=d.cpdag())
        self.assertEqual(c.arcs, {(0, 1), (0, 2)})
        self.assertEqual(c.edges, {(1, 2)})

    # def test_reversible_arcs(self):
    #     pass
    #
    # def test_shd(self):
    #     pass

    # def test_amat(self):
    #     amat, nodes = self.d.to_amat()
    #     for (i, j), val in np.ndenumerate(amat):
    #         if val == 1:
    #             self.assertTrue((nodes[i], nodes[j]) in self.d.arcs)
    #         elif val == 0:
    #             self.assertTrue((nodes[i], nodes[j]) not in self.d.arcs)

    def test_incident_arcs(self):
        self.assertEqual(self.d.incident_arcs(1), {(1, 2), (1, 3)})
        self.assertEqual(self.d.incident_arcs(2), {(1, 2), (2, 4)})
        self.assertEqual(self.d.incident_arcs(3), {(1, 3), (3, 4), (3, 5)})
        self.assertEqual(self.d.incident_arcs(4), {(2, 4), (3, 4)})
        self.assertEqual(self.d.incident_arcs(5), {(3, 5)})

    def test_shd(self):
        d1 = cd.DAG(arcs={(0, 1), (0, 2)})
        d2 = cd.DAG(arcs={(1, 0), (1, 2)})
        self.assertEqual(d1.shd(d2), 3)
        self.assertEqual(d2.shd(d1), 3)

        d1 = cd.DAG()
        d2 = cd.DAG(arcs={(0, 1), (1, 2)})
        self.assertEqual(d1.shd(d2), 2)
        self.assertEqual(d2.shd(d1), 2)

        d1 = cd.DAG(arcs={(0, 1), (1, 2)})
        d2 = cd.DAG(arcs={(0, 1), (2, 1)})
        self.assertEqual(d1.shd(d2), 1)
        self.assertEqual(d2.shd(d1), 1)

    def test_dsep(self):
        d = cd.DAG(arcs={(1, 2), (2, 3)})  # chain
        self.assertTrue(d.dsep(1, 3, {2}))
        self.assertFalse(d.dsep(1, 3))

        d = cd.DAG(arcs={(2, 1), (2, 3)})  # confounder
        self.assertTrue(d.dsep(1, 3, {2}))
        self.assertFalse(d.dsep(1, 3))

        d = cd.DAG(arcs={(1, 3), (2, 3)})  # v-structure
        self.assertTrue(d.dsep(1, 2))
        self.assertFalse(d.dsep(1, 2, {3}))

        d = cd.DAG(arcs={(1, 3), (2, 3), (3, 4), (4, 5)})  # v-structure with chain
        self.assertTrue(d.dsep(1, 2))
        self.assertFalse(d.dsep(1, 2, {5}))

    def test_is_invariant(self):
        d = cd.DAG(arcs={(1, 2), (2, 3)})
        self.assertTrue(d.is_invariant(1, 3))
        self.assertTrue(d.is_invariant(2, 3))
        self.assertFalse(d.is_invariant(3, 3))
        self.assertFalse(d.is_invariant(1, 3, cond_set=3))
        self.assertFalse(d.is_invariant(2, 3, cond_set=3))
        self.assertTrue(d.is_invariant(2, 3, cond_set=1))
        self.assertTrue(d.is_invariant(1, 3, cond_set=2))

    def test_marginal_mag(self):
        d = cd.DAG(arcs={(1, 2), (1, 3)})
        self.assertEqual(d.marginal_mag(1), cd.AncestralGraph(bidirected={(2, 3)}))

        d = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        self.assertEqual(d.marginal_mag(1), cd.AncestralGraph(directed={(2, 3)}))

    def test_markov_blanket(self):
        d = cd.DAG(arcs={(1, 2), (2, 3), (2, 4), (3, 5), (6, 3), (7, 4), (8, 4)})
        self.assertEqual(d.markov_blanket_of(2), {1, 3, 4, 6, 7, 8})

    # def test_vstructs(self):
    #     pass
    #
    # def test_to_cpdag(self):
    #     pass


if __name__ == '__main__':
    unittest.main()
