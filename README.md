**This package is nearing a V1 release, with potential (minor) breaking changes. After the release, future breaking changes will occur less frequently and with more notice. Please raise issues as needed.**

`causaldag` is common wrapper for the following packages:
* https://github.com/uhlerlab/graphical_models
* https://github.com/uhlerlab/conditional_independence
* https://github.com/uhlerlab/graphical_model_learning

Installing and importing `causaldag` should be sufficient for most use cases.

CausalDAG is a Python package for the creation, manipulation, and learning
of Causal DAGs. CausalDAG requires Python 3.5+

### Install
Install the latest version of CausalDAG:
```
$ pip3 install causaldag
```

### Documentation
Documentation for each subpackage is available at:
* graphical_models: https://graphical-models.readthedocs.io/en/latest/
* graphical_model_learning: https://graphical-model-learning.readthedocs.io/en/latest/
* conditional_independence: https://conditional-independence.readthedocs.io/en/latest/

Examples for specific algorithms can be found at https://uhlerlab.github.io/causaldag/

### Simple Example
Find the CPDAG (complete partially directed acyclic graph,
AKA the *essential graph*) corresponding to a DAG:
```
>>> from causaldag import rand, partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester, gsp
>>> import numpy as np
>>> np.random.seed(12312)
>>> nnodes = 5
>>> nodes = set(range(nnodes))
>>> dag = rand.directed_erdos(nnodes, .5)
>>> gdag = rand.rand_weights(dag)
>>> samples = gdag.sample(100)
>>> suffstat = partial_correlation_suffstat(samples)
>>> ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)
>>> est_dag = gsp(nodes, ci_tester)
>>> dag.shd_skeleton(est_dag)
```

### License

Released under the 3-Clause BSD license (see LICENSE.txt):
```
Copyright (C) 2018
Chandler Squires <csquires@mit.edu>
```
