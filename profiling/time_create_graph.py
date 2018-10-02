import causaldag as cd
import networkx as nx
from profiling.time_dec import timed


@timed
def test_create_nx_small():
    for _ in range(10000):
        g = nx.DiGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(1, 3)


@timed
def test_create_dag_small():
    for _ in range(10000):
        g = cd.DAG()
        g.add_arc(1, 2)
        g.add_arc(2, 3)
        g.add_arc(1, 3)


test_create_nx_small()
test_create_dag_small()

import numpy as np
np.random.seed(1729)
nnodes_large = 1000
arcs = cd.rand.directed_erdos(nnodes_large, .5).arcs
print(len(arcs))


@timed
def test_create_nx_large():
    for i in range(10):
        print(i)
        g = nx.DiGraph()
        g.add_nodes_from(range(nnodes_large))
        g.add_edges_from(arcs)


@timed
def test_create_dag_large():
    for i in range(10):
        print(i)
        g = cd.DAG(nodes=range(nnodes_large), arcs=arcs)


@timed
def test_diff():
    for i in range(5):
        gs = [cd.rand.directed_erdos(200, .5) for _ in range(10)]


@timed
def test_all_at_once():
    for i in range(5):
        gs = cd.rand.directed_erdos(200, .5, 10)


test_diff()
test_all_at_once()

