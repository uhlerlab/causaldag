"""
CausalDAG
=========

CausalDAG is a Python package for the creation, manipulation, and learning of Causal DAGs.

Simple Example
--------------

>>> import causaldag as cd
>>> dag = cd.DAG(arcs={(1, 2), (2, 3), (1, 3)})
>>> cpdag = dag.cpdag()
>>> iv = dag.optimal_intervention(cpdag=cpdag)
>>> icpdag = dag.interventional_cpdag([iv], cpdag=cpdag)
"""

from .classes import *
from .loaders import *

