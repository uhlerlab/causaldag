# Author: Chandler Squires
from unittest import TestCase
import unittest
import causaldag as cd


class TestDAG(TestCase):
    def test_cpdag_confounding(self):
        dag = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        cpdag = dag.cpdag()


if __name__ == '__main__':
    unittest.main()



