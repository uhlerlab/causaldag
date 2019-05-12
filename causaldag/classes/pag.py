import numpy as np


class PAG:
    def __init__(self, nodes, adjacencies, arrowheads, tails):
        self._nodes = nodes.copy()
        self._adjacencies = adjacencies.copy()
        self._arrowheads = arrowheads.copy()
        self._tails = tails.copy()

    def __eq__(self, other):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    @classmethod
    def from_amat(cls, amat):
        p = amat.shape[0]
        adjacencies = set()
        arrowheads = set()
        tails = set()
        for (i, j), val in np.ndenumerate(amat):
            if val == 1:
                adjacencies.add(frozenset({i, j}))
            if val == 2:
                arrowheads.add((i, j))
            if val == 3:
                tails.add((i, j))
        return PAG(set(range(p)), adjacencies, arrowheads, tails)

    def to_amat(self):
        raise NotImplementedError

    def add_adjacency(self, i, j):
        """Add the adjacency (i, j)
        """
        pass

    def add_arrowhead(self, i, j):
        """Add an arrowhead at j to the adjacency (i, j)
        """
        pass

    def add_tail(self, i, j):
        """Add a tail at i to the adjacency (i, j)
        """
        pass

    @property
    def nodes(self):
        return set(self._nodes)

    @property
    def adjacencies(self):
        return set(self._adjacencies)

    @property
    def arrowheads(self):
        return set(self._arrowheads)

    @property
    def tails(self):
        return set(self._tails)

    @property
    def circles(self):
        raise NotImplementedError

    @property
    def skeleton(self):
        return self._adjacencies.copy()

    def shd_skeleton(self, other):
        return len(self.skeleton.symmetric_difference(other.skeleton))

