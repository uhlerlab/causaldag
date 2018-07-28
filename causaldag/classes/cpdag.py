# Author: Chandler Squires
"""
Base class for completed partially directed acyclic graphs,
representing the Markov (and I-Markov) equivalence class of
DAGs
"""

from collections import defaultdict
from ..utils import core_utils
import itertools as itr
import numpy as np


class CPDAG:
    def __init__(self, dag_or_cpdag, known_edges=set()):
        self._nodes = dag_or_cpdag.nodes
        self._arcs = dag_or_cpdag.arcs
        self._parents = defaultdict(set, dag_or_cpdag.parents)
        self._children = defaultdict(set, dag_or_cpdag.children)
        self._neighbors = defaultdict(set, dag_or_cpdag.neighbors)
        self._edges = set()
        self._undirected_neighbors = defaultdict(set)

        from .dag import DAG
        if isinstance(dag_or_cpdag, DAG):
            self._protected = dag_or_cpdag.vstructs() | known_edges
        elif isinstance(dag_or_cpdag, CPDAG):
            self._protected = dag_or_cpdag._protected | known_edges

        self._replace_unprotected()

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

    def _replace_arc_with_edge(self, arc):
        self._arcs.remove(arc)
        self._edges.add(tuple(sorted(arc)))
        i, j = arc
        self._parents[j].remove(i)
        self._children[i].remove(j)
        self._undirected_neighbors[i].add(j)
        self._undirected_neighbors[j].add(i)

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

    def _replace_unprotected(self, verbose=False):
        PROTECTED = 'P'  # indicates that some configuration definitely exists to protect the edge
        UNDECIDED = 'U'  # indicates that some configuration exists that could protect the edge
        NOT_PROTECTED = 'N'  # indicates no possible configuration that could protect the edge

        undecided_edges = self._arcs - self._protected
        edge_flags = {edge: PROTECTED for edge in self._protected}
        edge_flags.update({edge: UNDECIDED for edge in undecided_edges})

        while undecided_edges:
            for edge in undecided_edges:
                i, j = edge
                flag = NOT_PROTECTED

                # check configuration (a) -- causal chain
                for k in self._parents[i]:
                    if not self.has_edge_or_arc(k, j):
                        if edge_flags[(k, i)] == PROTECTED:
                            flag = PROTECTED
                            break
                        else:
                            flag = UNDECIDED
                        if verbose: print('{edge} marked {flag} by (a)'.format(edge=edge, flag=flag))

                # check configuration (b) -- acyclicity
                if flag != PROTECTED:
                    for k in self._parents[j]:
                        if i in self._parents[k]:
                            if edge_flags[(i, k)] == PROTECTED and edge_flags[(k, j)] == PROTECTED:
                                flag = PROTECTED
                                break
                            else:
                                flag = UNDECIDED
                            if verbose: print('{edge} marked {flag} by (b)'.format(edge=edge, flag=flag))

                # check configuration (d)
                if flag != PROTECTED:
                    for k1, k2 in itr.combinations(self._parents[j], 2):
                        if self.has_edge(i, k1) and self.has_edge(i, k2) and not self.has_edge_or_arc(k1, k2):
                            if edge_flags[(k1, j)] == PROTECTED and edge_flags[(k2, j)] == PROTECTED:
                                flag = PROTECTED
                            else:
                                flag = UNDECIDED
                            if verbose: print('{edge} marked {flag} by (c)'.format(edge=edge, flag=flag))

                edge_flags[edge] = flag

            for edge in undecided_edges.copy():
                if edge_flags[edge] != UNDECIDED:
                    undecided_edges.remove(edge)
                if edge_flags[edge] == NOT_PROTECTED:
                    self._replace_arc_with_edge(edge)

    def interventional_cpdag(self, dag, intervened_nodes):
        cut_edges = set()
        for node in intervened_nodes:
            cut_edges.update(dag.incident_arcs(node))
        known_edges = cut_edges | self._protected
        return CPDAG(dag, known_edges=known_edges)

    def add_known_arc(self, i, j):
        return CPDAG(self, {(i, j)})

    def add_known_arcs(self, edges):
        return CPDAG(self, edges)

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

    @property
    def sources(self):
        # TODO: check logic - maybe should be *possible* sources
        return {node for node in self._nodes if len(self._parents) == 0}

    @property
    def sinks(self):
        # TODO: check logic - maybe should be *possible* sinks
        return {node for node in self._nodes if len(self._children[node]) == 0}

    def _neighbors_covered(self, node):
        return {node2: self.neighbors[node2] - {node} == self.neighbors[node] for node2 in self._nodes}

    def _all_dags(self):
        raise NotImplementedError
        sinks = self.sinks
        for sink in sinks:
            pass

    def all_dags(self):
        # pdag2alldags from pcalg.R
        raise NotImplementedError





