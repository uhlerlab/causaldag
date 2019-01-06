"""
CausalDAG
=========

CausalDAG is a Python package for the creation, manipulation, and learning of Causal DAGs.

Simple Example
--------------

>>> import causaldag as cd
>>> dag = cd.DAG(arcs={(1, 2), (2, 3), (1, 3)})
>>> cpdag = dag.cpdag()
>>> iv = dag.optimal_intervention_greedy(cpdag=cpdag)
>>> icpdag = dag.interventional_cpdag([iv], cpdag=cpdag)
{(1,2), (2,3)}

License
-------
Released under the 3-Clause BSD license::
   Copyright (C) 2018
   Chandler Squires <chandlersquires18@gmail.com>
"""

from .classes import *
from .loaders import *
from . import rand
from . import inference
from . import utils