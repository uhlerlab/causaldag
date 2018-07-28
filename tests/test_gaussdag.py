from unittest import TestCase
import unittest
import numpy as np
import causaldag as cd


class TestDAG(TestCase):
    def setUp(self):
        w = np.zeros((3, 3))
        w[0, 1] = 1
        w[0, 2] = -1
        w[1, 2] = 4
        self.gdag = cd.GaussDAG(weight_mat=w)

    def test_arcs(self):
        self.assertEqual(self.gdag.arcs, {(0, 1), (0, 2), (1, 2)})

    def test_add_node(self):
        pass

    def test_add_arc(self):
        pass

    def test_remove_node(self):
        pass

    def test_remove_arc(self):
        pass




if __name__ == '__main__':
    unittest.main()
