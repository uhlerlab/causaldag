CausalDAG is a Python package for the creation, manipulation, and learning
of Causal DAGs. CausalDAG requires Python 3.5+

### Install
Install the latest version of CausalDAG:
```
$ pip3 install causaldag
```

### Simple Example
Find the CPDAG (complete partially directed acyclic graph,
AKA the *essential graph*) corresponding to a DAG:
```
import causaldag as cd
dag = cd.DAG(arcs={(1, 2), (2, 3), (1, 3)})
cpdag = dag.cpdag()
```

### License
```
Chandler Squires <csquires@mit.edu>
```