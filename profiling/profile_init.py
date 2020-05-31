import causaldag as cd
from line_profiler import LineProfiler
from tqdm import tqdm

nnodes = 500
ngraphs = 100
dags = cd.rand.directed_erdos(nnodes, 40/(nnodes-1), ngraphs)
cpdags1 = [dag.cpdag_new(new=True) for dag in dags]
arcs_edges = [(cpdag.arcs, cpdag.edges) for cpdag in cpdags1]

print('add consecutively')
pdags = list(tqdm((cd.PDAG(arcs=arcs, edges=edges, new=False) for arcs, edges in arcs_edges), total=ngraphs))
print('add all at once')
pdags2 = list(tqdm((cd.PDAG(arcs=arcs, edges=edges, new=True) for arcs, edges in arcs_edges), total=ngraphs))
print('add consecutively')
pdags = list(tqdm((cd.PDAG(arcs=arcs, edges=edges, new=False) for arcs, edges in arcs_edges), total=ngraphs))
eq = [p == p2 for p, p2 in zip(pdags, pdags2)]
a = [p.num_edges for p in pdags]


def init_new():
    return list(tqdm((cd.PDAG(arcs=arcs, edges=edges, new=True) for arcs, edges in arcs_edges), total=ngraphs))


def init():
    return list(tqdm((cd.PDAG(arcs=arcs, edges=edges, new=False) for arcs, edges in arcs_edges), total=ngraphs))


lp = LineProfiler()
NEW = False
if NEW:
    lp.add_function(cd.PDAG._add_arcs_from)
    lp.runcall(init_new)
else:
    lp.add_function(cd.PDAG._add_arc)
    lp.runcall(init)
lp.print_stats()



# cpdags1 = list(tqdm((dag.cpdag() for dag in dags), total=ngraphs))
# equal = [cpdag1 == cpdag2 for cpdag1, cpdag2 in zip(cpdags1, cpdags2)]
# print(all(equal))
