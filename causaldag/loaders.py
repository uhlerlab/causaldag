from .classes.dag import DAG
import numpy as np


def load_gml(filename):
    raise NotImplementedError
    pass


def from_amat(amat):
    arcs = set()
    for (i, j), val in np.ndenumerate(amat):
        if val != 0:
            arcs.add((i, j))
    return DAG(arcs=arcs)


