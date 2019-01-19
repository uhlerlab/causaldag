from line_profiler import LineProfiler
import causaldag as cd
from causaldag.inference.structural import igsp
from causaldag.utils.ci_tests import gauss_ci_test, hsic_invariance_test
import numpy as np
import random
np.random.seed(1729)
random.seed(1729)

nnodes = 15
g = cd.rand.rand_weights(cd.rand.directed_erdos(nnodes, 3/(nnodes-1), 1))
iv_node = random
nsamples = 20
samples = {
    frozenset(): g.sample(nsamples),
    frozenset({iv_node}): g.sample_interventional_perfect({iv_node: cd.GaussIntervention(1, .1)}, nsamples)
}
corr = np.corrcoef(samples[frozenset()], rowvar=False)
suffstat = dict(C=corr, n=nsamples)
profiler = LineProfiler()


def run_igsp():
    for i in range(100):
        igsp(samples, suffstat, nnodes, gauss_ci_test, hsic_invariance_test, nruns=10)

profiler.add_function(igsp)
profiler.runcall(run_igsp)
profiler.print_stats()
