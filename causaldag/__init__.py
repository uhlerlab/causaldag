"""
CausalDAG
=========

CausalDAG is a Python package for the creation, manipulation, and learning of Causal DAGs.

Simple Example
--------------

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
3

License
-------
Released under the 3-Clause BSD license::
   Copyright (C) 2018
   Chandler Squires <chandlersquires18@gmail.com>
"""

# from .loaders import *
# from . import utils
from conditional_independence import *
from graphical_models import *
import graphical_models.rand as rand
from graphical_model_learning import *
