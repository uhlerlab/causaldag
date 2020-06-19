---
layout: default
title: UT-IGSP (Unknown-Target Interventional GSP)
description: Learning causal graphs from interventional data with unknown (or partially known) intervention targets.
---

The UT-IGSP algorithm learns a causal graph from interventional data. UT-IGSP is implemented as a part of the [causaldag](https://github.com/uhlerlab/causaldag) package. The source code
for UT-IGSP can be found in the package [here](https://github.com/uhlerlab/causaldag/blob/master/causaldag/structure_learning/dag/gsp.py). UT-IGSP is described in our paper, [Permutation-Based Causal Structure Learning
with Unknown Intervention Targets](https://arxiv.org/pdf/1910.09007.pdf).


## Install
To install the causaldag package:
```
$ pip3 install causaldag
```

## Simple Example
```python
from causaldag import unknown_target_igsp
import causaldag as cd
import random
from causaldag.utils.ci_tests import gauss_ci_suffstat, gauss_ci_test, MemoizedCI_Tester
from causaldag.utils.invariance_tests import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester

# Generate a random graph
nnodes = 10
nodes = set(range(nnodes))
exp_nbrs = 2
d = cd.rand.directed_erdos(nnodes, exp_nbrs/(nnodes-1))
g = cd.rand.rand_weights(d)

# Choose random intervention targets
num_targets = 2
num_settings = 2
targets_list = [random.sample(nodes, num_targets) for _ in range(num_settings)]
print(targets_list)

# Generate observational data
nsamples_obs = 1000
obs_samples = g.sample(nsamples_obs)

# Generate interventional data
iv_mean = 1
iv_var = .1
nsamples_iv = 1000
ivs = [{target: cd.GaussIntervention(iv_mean, iv_var) for target in targets} for targets in targets_list]
iv_samples_list = [g.sample_interventional(iv, nsamples_iv) for iv in ivs]

# Form sufficient statistics
obs_suffstat = gauss_ci_suffstat(obs_samples)
invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)


alpha = 1e-3
alpha_inv = 1e-3
ci_tester = MemoizedCI_Tester(gauss_ci_test, obs_suffstat, alpha=alpha)
invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

# Run UT-IGSP
setting_list = [dict(known_interventions=[]) for _ in targets_list]
est_dag, est_targets_list = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
print(est_targets_list)
```

## [Tutorial](./utigsp_tutorial.html)

## Code base
A codebase for reproducing the results of our paper can be found [here](https://github.com/csquires/utigsp).


[back](./)
