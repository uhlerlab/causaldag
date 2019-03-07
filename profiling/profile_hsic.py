from line_profiler import LineProfiler
import causaldag as cd
import random
import numpy as np
from causaldag.utils.ci_tests import hsic_test, gauss_ci_test

nnodes = 10
nsamples = 1000
nruns = 30

g = cd.rand.directed_erdos(nnodes, .5)
g = cd.rand.rand_weights(g)
samples = g.sample(nsamples)
corr = np.corrcoef(samples, rowvar=False)
suffstat = dict(C=corr, n=nsamples)


def run_hsic_test():
    for _ in range(nruns):
        i, j = random.sample(list(range(nnodes)), 2)
        cond_set = random.sample(set(range(nnodes)) - {i, j}, 2)
        hsic_test(samples, i, j, cond_set)


profiler = LineProfiler()
profiler.add_function(hsic_test)
profiler.runcall(run_hsic_test)
profiler.print_stats()


def run_gauss_ci_test():
    for _ in range(nruns):
        i, j = random.sample(list(range(nnodes)), 2)
        cond_set = set(random.sample(set(range(nnodes)) - {i, j}, 2))
        gauss_ci_test(suffstat, i, j, cond_set)


profiler = LineProfiler()
profiler.add_function(gauss_ci_test)
profiler.runcall(run_gauss_ci_test)
profiler.print_stats()
