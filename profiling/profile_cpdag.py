import causaldag as cd
from line_profiler import LineProfiler
from tqdm import tqdm

nnodes = 100
ngraphs = 100
dags = cd.rand.directed_erdos(nnodes, 40/(nnodes-1), ngraphs)
print('to undirected, then to_complete_pdag_new')
cpdags1 = list(tqdm((dag.cpdag_new(new=True) for dag in dags), total=ngraphs))
print('to undirected, then to_complete_pdag')
cpdag2 = list(tqdm((dag.cpdag_new(new=False) for dag in dags), total=ngraphs))
print('remove unprotected')
cpdag3 = list(tqdm((dag.cpdag() for dag in dags), total=ngraphs))


def compute_cpdags_new():
    return list(tqdm((dag.cpdag_new(new=True) for dag in dags), total=ngraphs))


def compute_cpdags():
    return list(tqdm((dag.cpdag_new(new=False) for dag in dags), total=ngraphs))


lp = LineProfiler()
NEW = True
if NEW:
    lp.add_function(cd.DAG.cpdag_new)
    lp.runcall(compute_cpdags_new)
else:
    lp.add_function(cd.DAG.cpdag_new)
    lp.runcall(compute_cpdags)
lp.print_stats()




# cpdags1 = list(tqdm((dag.cpdag() for dag in dags), total=ngraphs))
# equal = [cpdag1 == cpdag2 for cpdag1, cpdag2 in zip(cpdags1, cpdags2)]
# print(all(equal))
