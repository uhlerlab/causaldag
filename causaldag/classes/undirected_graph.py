from collections import defaultdict
from copy import deepcopy
import numpy as np
# from cvxopt import spmatrix


class UndirectedGraph:
    def __init__(self, nodes=set(), edges=set()):
        self._nodes = set(nodes)
        self._edges = {frozenset({i, j}) for i, j in edges}
        self._neighbors = defaultdict(set)
        self._degrees = defaultdict(int)
        for i, j in self._edges:
            self._nodes.add(i)
            self._nodes.add(j)
            self._neighbors[i].add(j)
            self._neighbors[j].add(i)
            self._degrees[i] += 1
            self._degrees[j] += 1

    def __eq__(self, other):
        return self._nodes == other._nodes and self._edges == other._edges

    @property
    def num_nodes(self):
        return len(self._nodes)

    def to_amat(self, sparse=False):
        """
        TODO

        Parameters
        ----------
        TODO

        Examples
        --------
        TODO
        """
        if sparse:
            raise NotImplementedError
            # js, ks = [], []
            # for j, k in self._edges:
            #     js.append(j)
            #     ks.append(k)
            #     js.append(k)
            #     ks.append(j)
            # return spmatrix(1, js, ks)
        amat = np.zeros([self.num_nodes, self.num_nodes], dtype=int)
        for i, j in self._edges:
            amat[i, j] = True
            amat[j, i] = True
        return amat

    @classmethod
    def from_amat(self, amat):
        """
        TODO

        Parameters
        ----------
        TODO

        Examples
        --------
        TODO
        """
        edges = {(i, j) for (i, j), val in np.ndenumerate(amat) if val != 0}
        return UndirectedGraph(nodes=set(range(amat.shape[0])), edges=edges)

    def copy(self, new=True):
        return UndirectedGraph(self._nodes, self._edges)

    @property
    def degrees(self):
        return {node: self._degrees[node] for node in self._nodes}

    @property
    def neighbors(self):
        return {node: self._neighbors[node] for node in self._nodes}

    @property
    def edges(self):
        return self._edges.copy()

    @property
    def nodes(self):
        return self._nodes.copy()

    @property
    def skeleton(self):
        return self.edges

    def has_edge(self, i, j):
        """
        TODO

        Parameters
        ----------
        TODO

        Examples
        --------
        TODO
        """
        return frozenset({i, j}) in self._edges

    def neighbors_of(self, node):
        """
        TODO

        Parameters
        ----------
        TODO

        Examples
        --------
        TODO
        """
        return self._neighbors[node].copy()

    # === MUTATORS ===
    def add_edge(self, i, j):
        """
        TODO

        Parameters
        ----------
        TODO

        Examples
        --------
        TODO
        """
        if frozenset({i, j}) not in self._edges:
            self._edges.add(frozenset({i, j}))
            self._neighbors[i].add(j)
            self._neighbors[j].add(i)
            self._degrees[i] += 1
            self._degrees[j] += 1

    def add_edges_from(self, edges):
        """
        TODO

        Parameters
        ----------
        TODO

        Examples
        --------
        TODO
        """
        for i, j in edges:
            self.add_edge(i, j)

    def delete_edges_from(self, edges):
        """
        TODO

        Parameters
        ----------
        TODO

        Examples
        --------
        TODO
        """
        for i, j in edges:
            self.delete_edge(i, j)

    def delete_edge(self, i, j):
        """
        TODO

        Parameters
        ----------
        TODO

        Examples
        --------
        TODO
        """
        self._edges.remove(frozenset({i, j}))
        self._neighbors[i].remove(j)
        self._neighbors[j].remove(i)
        self._degrees[i] -= 1
        self._degrees[j] -= 1

    def delete_node(self, i):
        """
        TODO

        Parameters
        ----------
        TODO

        Examples
        --------
        TODO
        """
        self._nodes.remove(i)
        for j in self._neighbors[i]:
            self._neighbors[j].remove(i)
            self._degrees[j] -= 1
            self._edges.remove(frozenset({i, j}))
        self._neighbors.pop(i, None)
        self._degrees.pop(i, None)

