from line_profiler import LineProfiler
import causaldag as cd
from causaldag.inference.structural import pcalg, skeleton
import numpy as np
from causaldag.utils.ci_tests import MemoizedCI_Tester, gauss_ci_suffstat, gauss_ci_test
import random
np.random.seed(1729)
random.seed(1729)

nnodes = 20
nodes = set(range(nnodes))
g = cd.rand.rand_weights(cd.rand.directed_erdos(nnodes, 3/(nnodes-1), 1))
iv_node = random
nsamples = 1000
samples = g.sample(nsamples)
suffstat = gauss_ci_suffstat(samples)
profiler = LineProfiler()


def run_pc():
    for i in range(100):
        ci_tester = MemoizedCI_Tester(gauss_ci_test, suffstat)
        pcalg(nodes, ci_tester, max_cond_set=None, verbose=True)


profiler.add_function(pcalg)
profiler.runcall(run_pc)
profiler.print_stats()
