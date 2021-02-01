**This package is nearing a V1 release, with potential (minor) breaking changes. After the release, future breaking changes will occur less frequently and with more notice. Please raise issues as needed.**

The functionalities of the package have been divided into three sub-packages:
* https://github.com/uhlerlab/graphical_models
* https://github.com/uhlerlab/conditional_independence
* https://github.com/uhlerlab/graphical_model_learning

However, the `causaldag` package will be a common wrapper for user convenience, and hence installing and importing `causaldag` should be sufficient for most use cases.

[![codecov](https://codecov.io/gh/uhlerlab/causaldag/branch/master/graph/badge.svg?token=RSM00FKU9A)](https://codecov.io/gh/uhlerlab/causaldag)

CausalDAG is a Python package for the creation, manipulation, and learning
of Causal DAGs. CausalDAG requires Python 3.5+

### Install
Install the latest version of CausalDAG:
```
$ pip3 install causaldag
```

### Documentation
Documentation is available at https://causaldag.readthedocs.io/en/latest/index.html

Examples for specific algorithms can be found at https://uhlerlab.github.io/causaldag/

### Simple Example
Find the CPDAG (complete partially directed acyclic graph,
AKA the *essential graph*) corresponding to a DAG:
```
>>> import causaldag as cd
>>> dag = cd.DAG(arcs={(1, 2), (2, 3), (1, 3)})
>>> cpdag = dag.cpdag()
>>> iv = dag.optimal_intervention(cpdag=cpdag)
>>> icpdag = dag.interventional_cpdag([iv], cpdag=cpdag)
>>> dag.reversible_arcs()
{(1,2), (2,3)}
```

### License

Released under the 3-Clause BSD license (see LICENSE.txt):
```
Copyright (C) 2018
Chandler Squires <csquires@mit.edu>
```
