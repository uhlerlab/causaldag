import causaldag as cd
import numpy as np
from causaldag.inference.structural import gsp
from causaldag.utils.ci_tests import gauss_ci_test, GaussCIMemoizer
from line_profiler import LineProfiler


nnodes = 20
nsamples = 100

d = cd.rand.directed_erdos(nnodes, .5)
d = cd.rand.rand_weights(d)
samples = d.sample(nsamples)
corr = np.corrcoef(samples, rowvar=False)
m = GaussCIMemoizer(corr)

suffstat = dict(C=corr, n=nsamples)
c1 = m.ci_test(suffstat, 0, 1)

c2 = cd.utils.ci_tests.gauss_ci_test(suffstat, 0, 1)


def run_gsp_memo_ci():
    for i in range(20):
        m.partial_correlations = dict()
        gsp(suffstat, nnodes, m.ci_test, nruns=10)


def run_gsp_normal_ci():
    for i in range(20):
        gsp(suffstat, nnodes, gauss_ci_test, nruns=10)


# profiler = LineProfiler()
# profiler.add_function(m.ci_test)
# profiler.runcall(run_gsp_memo_ci)
# profiler.print_stats()

profiler = LineProfiler()
profiler.add_function(gauss_ci_test)
profiler.runcall(run_gsp_normal_ci)
profiler.print_stats()
