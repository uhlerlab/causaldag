import causaldag as cd
from causaldag.utils.ci_tests import MemoizedCI_Tester, gauss_ci_test, gauss_ci_suffstat
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

nnodes = 50
exp_nbrs_list = [2]*5 + [3]*5 + [4]*5 + [5]*5
ngraphs = len(exp_nbrs_list)
nsamples = 2*nnodes
dags = [cd.rand.directed_erdos(nnodes, exp_nbrs/(nnodes-1)) for exp_nbrs in exp_nbrs_list]
gdags = [cd.rand.rand_weights(dag) for dag in dags]
samples = [gdag.sample(nsamples) for gdag in gdags]
suffstats = [gauss_ci_suffstat(samples) for samples in samples]
ci_testers1 = [MemoizedCI_Tester(gauss_ci_test, suffstat) for suffstat in suffstats]

perms = [random.sample(list(range(nnodes)), nnodes) for _ in range(ngraphs)]
imaps1 = list(tqdm((cd.perm2dag(perm, ci_tester, verbose=False) for perm, ci_tester in zip(perms, ci_testers1)), total=ngraphs))
true_max_degrees = [dag.max_in_degree for dag in dags]
ci_tests_per_dag = [list(zip(*ci_tester.ci_dict.keys()))[-1] for ci_tester in ci_testers1]
ci_tests_sizes = [np.array([len(ci_test) for ci_test in ci_tests]) for ci_tests in ci_tests_per_dag]
max_ci_test_sizes = [sizes.max() for sizes in ci_tests_sizes]

plt.clf()
plt.scatter(true_max_degrees, max_ci_test_sizes)
plt.xlabel('Max degree')
plt.ylabel('Max test size')
# plt.ion()
# plt.show()
