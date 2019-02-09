**This package is still very young and hence maybe be subject to future revision.
Use at your own risk.**

CausalDAG is a Python package for the creation, manipulation, and learning
of Causal DAGs. CausalDAG requires Python 3.5+

### Install
Install the latest version of CausalDAG:
```
$ pip3 install causaldag
```

### Documentation
Documentation is available at https://causaldag.readthedocs.io/en/latest/index.html

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
