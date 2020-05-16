---
layout: default
title: DCI (Difference Causal Inference)
description: Learning differences of causal graphs.
---

The Difference Causal Inference (DCI) algorithm directly learns the difference between two causal graphs given two datasets. DCI is implemented as a part of the [causaldag](https://github.com/uhlerlab/causaldag) package. The source code
for DCI can be found in the package [here](https://github.com/uhlerlab/causaldag/blob/master/causaldag/structure_learning/difference/difference_dag.py).

![](images/dci.png)

## Install
To install the causaldag package:
```
$ pip3 install causaldag
```

## Simple Example
```python
from causaldag import dci, dci_stability_selection
from causaldag.datasets import create_synthetic_difference
import numpy as np
import itertools as itr

X1, X2, true_difference = create_synthetic_difference(nnodes=8, nsamples=10000)
p = X1.shape[1]

difference_matrix = dci(X1, X2, difference_ug=list(itr.combinations(range(p), 2)))
ddag_edges = set(zip(*np.where(difference_matrix != 0)))
print(true_difference)
print(ddag_edges)
```

## [Tutorial](./dci_tutorial.html)
DCI is applied to gene expression data collected from two different conditions to learn the causal difference gene regulatory network.


[back](./)