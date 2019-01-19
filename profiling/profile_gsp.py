from line_profiler import LineProfiler
import causaldag as cd
from causaldag.inference.structural import gsp
from causaldag.utils.ci_tests import gauss_ci_test
import numpy as np
import random
np.random.seed(1729)
random.seed(1729)

nnodes = 15
g = cd.rand.rand_weights(cd.rand.directed_erdos(nnodes, 3/(nnodes-1), 1))
iv_node = random
nsamples = 100
samples = {
    frozenset(): g.sample(nsamples),
    frozenset({iv_node}): g.sample_interventional_perfect({iv_node: cd.GaussIntervention(1, .1)}, nsamples)
}
corr = np.corrcoef(samples[frozenset()], rowvar=False)
suffstat = dict(C=corr, n=nsamples)
profiler = LineProfiler()


def run_gsp():
    for i in range(20):
        gsp(suffstat, nnodes, gauss_ci_test, nruns=10)

profiler.add_function(gsp)
profiler.runcall(run_gsp)
profiler.print_stats()
