# Author: Chandler Squires
"""
Base class for partially directed acyclic graphs
"""

from collections import defaultdict
from ..utils import core_utils
import itertools as itr
import numpy as np


class PDAG:
    def __init__(self, dag_or_pdag, known_arcs=set()):
        self._nodes = set(dag_or_pdag._nodes)
        self._arcs = set(dag_or_pdag._arcs)

        self._parents = defaultdict(set, dag_or_pdag.parents)
        self._children = defaultdict(set, dag_or_pdag.children)
        self._neighbors = defaultdict(set, dag_or_pdag.neighbors)

        from .dag import DAG
        if isinstance(dag_or_pdag, DAG):
            self._edges = set()
            self._known_arcs = dag_or_pdag.vstructs() | known_arcs
            self._undirected_neighbors = defaultdict(set)
        elif isinstance(dag_or_pdag, PDAG):
            self._edges = set(dag_or_pdag._edges)
            self._known_arcs = dag_or_pdag._known_arcs | known_arcs
            self._undirected_neighbors = defaultdict(set, dag_or_pdag.undirected_neighbors)

    def __eq__(self, other):
        same_nodes = self._nodes == other._nodes
        same_arcs = self._arcs == other._arcs
        same_edges = self._edges == other._edges

        return same_nodes and same_arcs and same_edges

    def __str__(self):
        substrings = []
        for node in self._nodes:
            parents = self._parents[node]
            nbrs = self._undirected_neighbors[node]
            parents_str = ','.join(map(str, parents)) if len(parents) != 0 else ''
            nbrs_str = ','.join(map(str, nbrs)) if len(nbrs) != 0 else ''

            if len(parents) == 0 and len(nbrs) == 0:
                substrings.append('[{node}]'.format(node=node))
            else:
                substrings.append('[{node}|{parents}:{nbrs}]'.format(node=node, parents=parents_str, nbrs=nbrs_str))
        return ''.join(substrings)

    def remove_node(self, node):
        self._nodes.remove(node)
        self._arcs = {(i, j) for i, j in self._arcs if i != node and j != node}
        self._edges = {(i, j) for i, j in self._edges if i != node and j != node}
        for child in self._children[node]:
            self._parents[child].remove(node)
            self._neighbors[child].remove(node)
        for parent in self._parents[node]:
            self._children[parent].remove(node)
            self._neighbors[parent].remove(node)
        for u_nbr in self._neighbors[node]:
            self._undirected_neighbors[u_nbr].remove(node)
        del self._parents[node]
        del self._children[node]
        del self._neighbors[node]
        del self._undirected_neighbors[node]

    @property
    def nodes(self):
        return set(self._nodes)

    @property
    def arcs(self):
        return set(self._arcs)

    @property
    def edges(self):
        return set(self._edges)

    @property
    def parents(self):
        return core_utils.defdict2dict(self._parents, self._nodes)

    @property
    def children(self):
        return core_utils.defdict2dict(self._children, self._nodes)

    @property
    def neighbors(self):
        return core_utils.defdict2dict(self._neighbors, self._nodes)

    @property
    def undirected_neighbors(self):
        return core_utils.defdict2dict(self._undirected_neighbors, self._nodes)

    def copy(self):
        return PDAG(self)

    def _replace_arc_with_edge(self, arc):
        self._arcs.remove(arc)
        self._edges.add(tuple(sorted(arc)))
        i, j = arc
        self._parents[j].remove(i)
        self._children[i].remove(j)
        self._undirected_neighbors[i].add(j)
        self._undirected_neighbors[j].add(i)

    def _replace_edge_with_arc(self, arc):
        self._edges.remove(tuple(sorted(arc)))
        self._arcs.add(arc)
        i, j = arc
        self._parents[j].add(i)
        self._children[i].add(j)
        self._undirected_neighbors[i].remove(j)
        self._undirected_neighbors[j].remove(i)

    def vstructs(self):
        vstructs = set()
        for node in self._nodes:
            for p1, p2 in itr.combinations(self._parents[node], 2):
                if p1 not in self._parents[p2] and p2 not in self._parents[p1]:
                    vstructs.add((p1, node))
                    vstructs.add((p2, node))
        return vstructs

    def has_edge(self, i, j):
        return (i, j) in self._edges or (j, i) in self._edges

    def has_edge_or_arc(self, i, j):
        return (i, j) in self._arcs or (j, i) in self._arcs or self.has_edge(i, j)

    def add_protected_orientations(self):
        pass

    def remove_unprotected_orientations(self, verbose=False):
        """
        Replace with edges those arcs whose orientations cannot be determined by either:
        - prior knowledge, or
        - Meek's rules

        =====
        See Koller & Friedman, Algorithm 3.5

        :param verbose:
        :return:
        """
        PROTECTED = 'P'  # indicates that some configuration definitely exists to protect the edge
        UNDECIDED = 'U'  # indicates that some configuration exists that could protect the edge
        NOT_PROTECTED = 'N'  # indicates no possible configuration that could protect the edge

        undecided_arcs = self._arcs - self._known_arcs
        arc_flags = {arc: PROTECTED for arc in self._known_arcs}
        arc_flags.update({arc: UNDECIDED for arc in undecided_arcs})

        while undecided_arcs:
            for arc in undecided_arcs:
                i, j = arc
                flag = NOT_PROTECTED

                # check configuration (a) -- causal chain
                for k in self._parents[i]:
                    if not self.has_edge_or_arc(k, j):
                        if arc_flags[(k, i)] == PROTECTED:
                            flag = PROTECTED
                            break
                        else:
                            flag = UNDECIDED
                        if verbose: print('{edge} marked {flag} by (a)'.format(edge=arc, flag=flag))

                # check configuration (b) -- acyclicity
                if flag != PROTECTED:
                    for k in self._parents[j]:
                        if i in self._parents[k]:
                            if arc_flags[(i, k)] == PROTECTED and arc_flags[(k, j)] == PROTECTED:
                                flag = PROTECTED
                                break
                            else:
                                flag = UNDECIDED
                            if verbose: print('{edge} marked {flag} by (b)'.format(edge=arc, flag=flag))

                # check configuration (d)
                if flag != PROTECTED:
                    for k1, k2 in itr.combinations(self._parents[j], 2):
                        if self.has_edge(i, k1) and self.has_edge(i, k2) and not self.has_edge_or_arc(k1, k2):
                            if arc_flags[(k1, j)] == PROTECTED and arc_flags[(k2, j)] == PROTECTED:
                                flag = PROTECTED
                            else:
                                flag = UNDECIDED
                            if verbose: print('{edge} marked {flag} by (c)'.format(edge=arc, flag=flag))

                arc_flags[arc] = flag

            for arc in undecided_arcs.copy():
                if arc_flags[arc] != UNDECIDED:
                    undecided_arcs.remove(arc)
                if arc_flags[arc] == NOT_PROTECTED:
                    self._replace_arc_with_edge(arc)

    def interventional_cpdag(self, dag, intervened_nodes):
        cut_edges = set()
        for node in intervened_nodes:
            cut_edges.update(dag.incident_arcs(node))
        known_edges = cut_edges | self._known_arcs
        return PDAG(dag, known_arcs=known_edges)

    def add_known_arc(self, i, j):
        if (i, j) in self._known_arcs:
            return
        self._known_arcs.add((i, j))
        self._edges.remove(tuple(sorted((i, j))))
        self.remove_unprotected_orientations()

    def add_known_arcs(self, arcs):
        raise NotImplementedError

    def to_amat(self, node_list=None):
        import pandas as pd

        if node_list is None:
            node_list = sorted(self._nodes)

        node2ix = {node: i for i, node in enumerate(node_list)}
        amat = np.zeros([len(self._nodes), len(self._nodes)], dtype=int)
        for source, target in self._arcs:
            amat[node2ix[source], node2ix[target]] = 1
        for i, j in self._edges:
            amat[node2ix[i], node2ix[j]] = 1
            amat[node2ix[j], node2ix[i]] = 1

        return pd.DataFrame(amat, index=node_list, columns=node_list)

    def _possible_sinks(self):
        return {node for node in self._nodes if len(self._children[node]) == 0}

    def _neighbors_covered(self, node):
        return {node2: self.neighbors[node2] - {node} == self.neighbors[node] for node2 in self._nodes}

    def to_dag(self):
        from .dag import DAG

        pdag2 = self.copy()
        arcs = set()
        while len(pdag2._edges) + len(pdag2._arcs) != 0:
            is_sink = lambda n: len(pdag2._children[n]) == 0
            adj_check = lambda n: all(
                (pdag2._neighbors[n] - {u_nbr}).issubset(pdag2._neighbors[u_nbr])
                for u_nbr in pdag2._undirected_neighbors[n]
            )
            sink = next(n for n in pdag2._nodes if is_sink(n) and adj_check(n))
            print(sink)
            if sink is None:
                break
            arcs.update((u_nbr, sink) for u_nbr in pdag2._undirected_neighbors[sink])
            pdag2.remove_node(sink)

        return DAG(arcs=arcs)

    def all_dags(self, verbose=False):
        amat = self.to_amat()
        node_list = list(amat.index)
        all_dags = set()
        _all_dags_helper(amat, amat, node_list, all_dags, verbose=verbose)
        return all_dags


def _all_dags_helper(full_amat, curr_submatrix, node_list, all_dags, verbose=False):
    if curr_submatrix.sum().sum() == 0:
        arcs = frozenset((node_list[i], node_list[j]) for (i, j), val in np.ndenumerate(full_amat) if val==1)
        all_dags.add(arcs)
        if verbose: print('=== APPENDING ===')
        if verbose: print(arcs)
        if verbose: print('=================')
        return

    if verbose: print(full_amat)
    nchildren = ((curr_submatrix - curr_submatrix.T) > 0).sum(axis=1)
    if verbose: print('nchildren\n', nchildren)
    sink_ixs = (nchildren == 0).nonzero()[0]
    sinks = curr_submatrix.index[sink_ixs]

    if verbose: print(set(sinks))
    for sink in sinks:
        children_ixs = curr_submatrix.loc[sink].nonzero()[0]
        children = set(curr_submatrix.index[children_ixs])
        parent_ixs = curr_submatrix[sink].nonzero()[0]
        parents = set(curr_submatrix.index[parent_ixs])

        undirected_nbrs = list(children & parents)
        sink_nbrs = children | parents
        get_neighbors = lambda n: set(curr_submatrix.index[curr_submatrix[n].nonzero()[0]]) | set(curr_submatrix.index[curr_submatrix[n].nonzero()[0]])

        nbrs_of_undirected_nbrs = (get_neighbors(nbr) for nbr in undirected_nbrs)
        nbrs_of_undirected_nbrs = list(nbrs_of_undirected_nbrs)
        if len(sink_nbrs) > 0 and all((sink_nbrs - {nbr}).issubset(nbrs_of_nbr) for nbr, nbrs_of_nbr in zip(undirected_nbrs, nbrs_of_undirected_nbrs)):
            if verbose: print('Removing sink node', sink, 'and edges to', sink_nbrs, 'from:\n', full_amat)
            new_full_amat = full_amat.copy()
            new_full_amat.loc[sink][sink_nbrs] = 0
            new_submatrix = curr_submatrix.drop(sink, axis=0).drop(sink, axis=1)
            _all_dags_helper(new_full_amat, new_submatrix, node_list, all_dags, verbose=verbose)



