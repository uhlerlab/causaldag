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
        if node_list is None:
            node_list = sorted(self._nodes)

        node2ix = {node: i for i, node in enumerate(node_list)}
        amat = np.zeros([len(self._nodes), len(self._nodes)])
        for source, target in self._arcs:
            amat[node2ix[source], node2ix[target]] = 1
        for i, j in self._edges:
            amat[node2ix[i], node2ix[j]] = 1
            amat[node2ix[j], node2ix[i]] = 1
        return amat, node_list

    def _possible_sinks(self):
        return {node for node in self._nodes if len(self._children[node]) == 0}

    def _neighbors_covered(self, node):
        return {node2: self.neighbors[node2] - {node} == self.neighbors[node] for node2 in self._nodes}

    def all_dags(self):
        all_dags = []
        _all_dags_helper(self, all_dags)
        return all_dags


def _all_dags_helper(curr_graph, curr_dags, sinked_nodes=None):
    if sinked_nodes is None:
        sinked_nodes = set()

    print(curr_graph)
    print(sinked_nodes)
    if len(curr_graph._edges) == 0:
        print('here')
        curr_dags.append(curr_graph)
        return

    sinks = {node for node in curr_graph._nodes if len(curr_graph._children[node] - sinked_nodes) == 0}
    for sink in sinks:
        undirected_nbrs = curr_graph._undirected_neighbors[sink]
        all_undirected_protected = all(
            (curr_graph._neighbors[sink] - {nbr}) == (curr_graph._neighbors[nbr] - {sink})
            for nbr in undirected_nbrs
        )
        if all_undirected_protected and len(undirected_nbrs) != 0:
            sinked_nodes = set(sinked_nodes)
            sinked_nodes.add(sink)

            new_graph = curr_graph.copy()
            for nbr in undirected_nbrs:
                new_graph._replace_edge_with_arc((nbr, sink))
            _all_dags_helper(new_graph, curr_dags, sinked_nodes)


