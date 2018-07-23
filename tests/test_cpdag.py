from unittest import TestCase
import unittest
import causaldag as cd


class TestDAG(TestCase):
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

    def test_interventional_cpdag(self):
        dag = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        cpdag = dag.cpdag()
        int_cpdag = dag.interventional_cpdag({1}, cpdag=cpdag)
        self.assertEqual(int_cpdag.arcs, {(1, 2), (1, 3)})
        self.assertEqual(int_cpdag.edges, {(2, 3)})
        self.assertEqual(int_cpdag.undirected_neighbors[1], set())
        self.assertEqual(int_cpdag.undirected_neighbors[2], {3})
        self.assertEqual(int_cpdag.undirected_neighbors[3], {2})

    def test_optimal_intervention(self):
        dag = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        self.assertEqual(dag.optimal_intervention(), 2)


if __name__ == '__main__':
    unittest.main()


