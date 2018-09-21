# Author: Chandler Squires
"""
Base class for causal DAGs
"""

from collections import defaultdict
import numpy as np
import itertools as itr
from ..utils import core_utils
import operator as op
from typing import Set


class CycleError(Exception):
    def __init__(self, cycle):
        self.cycle = cycle
        message = 'Adding arc(s) causes the cycle ' + path2str(cycle)
        super().__init__(message)


def path2str(path):
    return '->'.join(map(str, path))


class DAG:
    def __init__(self, nodes: Set=set(), arcs: Set=set()):
        self._nodes = nodes.copy()
        self._arcs = set()
        self._neighbors = defaultdict(set)
        self._parents = defaultdict(set)
        self._children = defaultdict(set)
        for i, j in arcs:
            self._add_arc(i, j)

    @classmethod
    def from_amat(cls, amat):
        nodes = set(range(amat.shape[0]))
        arcs = set()
        for (i, j), val in np.ndenumerate(amat):
            if val != 0:
                arcs.add((i, j))
        return DAG(nodes=nodes, arcs=arcs)

    def copy(self):
        return DAG(nodes=self._nodes, arcs=self._arcs)

    @property
    def nodes(self):
        return set(self._nodes)

    @property
    def arcs(self):
        return set(self._arcs)

    @property
    def neighbors(self):
        return core_utils.defdict2dict(self._neighbors, self._nodes)

    @property
    def parents(self):
        return core_utils.defdict2dict(self._parents, self._nodes)

    @property
    def children(self):
        return core_utils.defdict2dict(self._children, self._nodes)

    def has_arc(self, source, target):
        return (source, target) in self._arcs

    def __str__(self):
        t = self.topological_sort()
        substrings = []
        for node in t:
            if self._parents[node]:
                parents_str = ','.join(map(str, self._parents[node]))
                substrings.append('[%s|%s]' % (node, parents_str))
            else:
                substrings.append('[%s]' % node)
        return ''.join(substrings)

    # === MUTATORS
    def add_node(self, node):
        self._nodes.add(node)

    def add_arc(self, i, j):
        self._add_arc(i, j)
        try:
            self._check_acyclic()
        except CycleError as e:
            self.remove_arc(i, j)
            raise e

    def _check_acyclic(self):
        self.topological_sort()

    def _mark_children_visited(self, node, any_visited, curr_path_visited, curr_path, stack):
        any_visited[node] = True
        curr_path_visited[node] = True
        curr_path.append(node)
        for child in self._children[node]:
            if not any_visited[child]:
                self._mark_children_visited(child, any_visited, curr_path_visited, curr_path, stack)
            elif curr_path_visited[child]:
                cycle = curr_path + [child]
                raise CycleError(cycle)
        curr_path.pop()
        curr_path_visited[node] = False
        stack.append(node)

    def topological_sort(self):
        any_visited = {node: False for node in self._nodes}
        curr_path_visited = {node: False for node in self._nodes}
        curr_path = []
        stack = []
        for node in self._nodes:
            if not any_visited[node]:
                self._mark_children_visited(node, any_visited, curr_path_visited, curr_path, stack)
        return list(reversed(stack))

    def _add_arc(self, i, j):
        self._nodes.add(i)
        self._nodes.add(j)
        self._arcs.add((i, j))

        self._neighbors[i].add(j)
        self._neighbors[j].add(i)

        self._children[i].add(j)
        self._parents[j].add(i)

    def add_nodes_from(self, nodes):
        for node in nodes:
            self.add_node(node)

    def add_arcs_from(self, arcs):
        for i, j in arcs:
            self._add_arc(i, j)
        try:
            self._check_acyclic()
        except CycleError as e:
            for i, j in arcs:
                self.remove_arc(i, j)
            raise e

    def reverse_arc(self, i, j, ignore_error=False):
        try:
            self._arcs.remove((i, j))
            self._parents[j].remove(i)
            self._children[i].remove(j)

            self._arcs.add((j, i))
            self._parents[i].add(j)
            self._children[j].add(i)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_arc(self, i, j, ignore_error=False):
        try:
            self._arcs.remove((i, j))
            self._parents[j].remove(i)
            self._children[i].remove(j)
            self._neighbors[j].remove(i)
            self._neighbors[i].remove(j)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_node(self, node, ignore_error=False):
        try:
            self._nodes.remove(node)
            for parent in self._parents[node]:
                self._children[parent].remove(node)
                self._neighbors[parent].remove(node)
            for child in self._children[node]:
                self._parents[child].remove(node)
                self._neighbors[child].remove(node)
            del self._neighbors[node]
            del self._parents[node]
            del self._children[node]
            self._arcs = {(i, j) for i, j in self._arcs if i != node and j != node}

        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    # === GRAPH PROPERTIES
    def reversible_arcs(self):
        reversible_arcs = set()
        for i, j in self._arcs:
            if self._parents[i] == (self._parents[j] - {i}):
                reversible_arcs.add((i, j))
        return reversible_arcs

    def vstructs(self):
        vstructs = set()
        for node in self._nodes:
            for p1, p2 in itr.combinations(self._parents[node], 2):
                if p1 not in self._parents[p2] and p2 not in self._parents[p1]:
                    vstructs.add((p1, node))
                    vstructs.add((p2, node))
        return vstructs

    # === COMPARISON
    def shd(self, other) -> int:
        if isinstance(other, DAG):
            return len(other.arcs - self.arcs) + len(self.arcs - other.arcs)

    # === CONVENIENCE
    def _add_downstream(self, downstream, node):
        for child in self._children[node]:
            if child not in downstream:
                downstream.add(child)
                self._add_downstream(downstream, child)

    def downstream(self, node):
        downstream = set()
        self._add_downstream(downstream, node)
        return downstream

    def _add_upstream(self, upstream, node):
        for parent in self._parents[node]:
            if parent not in upstream:
                upstream.add(parent)
                self._add_upstream(upstream, parent)

    def upstream(self, node):
        upstream = set()
        self._add_upstream(upstream, node)
        return upstream

    def incident_arcs(self, node):
        incident_arcs = set()
        for child in self._children[node]:
            incident_arcs.add((node, child))
        for parent in self._parents[node]:
            incident_arcs.add((parent, node))
        return incident_arcs

    def incoming_arcs(self, node):
        incoming_arcs = set()
        for parent in self._parents[node]:
            incoming_arcs.add((parent, node))
        return incoming_arcs

    def outgoing_arcs(self, node):
        outgoing_arcs = set()
        for child in self._children[node]:
            outgoing_arcs.add((node, child))
        return outgoing_arcs

    def outdegree(self, node):
        return len(self._children[node])

    def indegree(self, node):
        return len(self._parents[node])

    # === CONVERTERS
    def save_gml(self, filename):
        tab = '  '
        indent = 0
        newline = lambda indent: '\n' + (tab * indent)
        with open(filename, 'w') as f:
            f.write('graph [')
            indent += 1
            f.write(newline(indent))
            f.write('directed 1')
            f.write(newline(indent))
            node2ix = core_utils.ix_map_from_list(self._nodes)
            for node, ix in node2ix.items():
                f.write('node [')
                indent += 1
                f.write(newline(indent))
                f.write('id %s' % ix)
                f.write(newline(indent))
                f.write('label "%s"' % node)
                indent -= 1
                f.write(newline(indent))
                f.write(']')
                f.write(newline(indent))
            for source, target in self._arcs:
                f.write('edge [')
                indent += 1
                f.write(newline(indent))
                f.write('source %s' % source)
                f.write(newline(indent))
                f.write('target %s' % target)
                indent -= 1
                f.write(newline(indent))
                f.write(']')
                f.write(newline(indent))
            f.write(']')

    def to_amat(self, node_list=None, mode='dataframe'):
        if not node_list:
            node_list = sorted(self._nodes)
        node2ix = {node: i for i, node in enumerate(node_list)}

        shape = (len(self._nodes), len(self._nodes))
        if mode == 'dataframe' or mode == 'numpy':
            amat = np.zeros(shape, dtype=int)
        else:
            from scipy.sparse import lil_matrix
            amat = lil_matrix(shape, dtype=int)

        for source, target in self._arcs:
            amat[node2ix[source], node2ix[target]] = 1

        if mode == 'dataframe':
            from pandas import DataFrame
            return DataFrame(amat, index=node_list, columns=node_list)
        else:
            return amat, node_list

    # === optimal interventions
    def cpdag(self):
        from .pdag import PDAG
        pdag = PDAG(nodes=self._nodes, arcs=self._arcs, known_arcs=self.vstructs())
        pdag.remove_unprotected_orientations()
        return pdag

    def interventional_cpdag(self, intervened_nodes, cpdag=None):
        from .pdag import PDAG

        if cpdag is None:
            dag_cut = self.copy()
            known_arcs = set()
            for node in intervened_nodes:
                for i, j in dag_cut.incoming_arcs(node):
                    dag_cut.remove_arc(i, j)
                    known_arcs.update(self.outgoing_arcs(node))
            known_arcs.update(dag_cut.vstructs())
            pdag = PDAG(dag_cut._nodes, dag_cut._arcs, known_arcs=known_arcs)
        else:
            cut_edges = set()
            for node in intervened_nodes:
                cut_edges.update(self.incident_arcs(node))
            known_arcs = cut_edges | cpdag._known_arcs
            pdag = PDAG(self._nodes, self._arcs, known_arcs=known_arcs)

        pdag.remove_unprotected_orientations()
        return pdag

    def optimal_intervention_greedy(self, cpdag=None, num_interventions=1):
        if cpdag is None:
            cpdag = self.cpdag()
        if len(cpdag.edges) == 0:
            return [None]*num_interventions, [cpdag]*num_interventions

        max_one_undirected_nbr = all(len(cpdag._undirected_neighbors[node]) <= 1 for node in self._nodes)
        no_undirected_nbrs = lambda node: cpdag._undirected_neighbors[node] == 0
        better_neighbor = lambda node: len(cpdag._undirected_neighbors[node]) == 1 and not max_one_undirected_nbr
        considered_nodes = list(filter(lambda node: not (no_undirected_nbrs(node) or better_neighbor(node)), self._nodes))

        nodes2icpdags = {
            node: self.interventional_cpdag({node}, cpdag=cpdag)
            for node in considered_nodes
        }
        nodes2num_oriented = {
            node: len(icpdag._arcs)
            for node, icpdag in nodes2icpdags.items()
        }

        best_iv = max(nodes2num_oriented.items(), key=op.itemgetter(1))[0]
        icpdag = nodes2icpdags[best_iv]

        if num_interventions == 1:
            return [best_iv], [icpdag]
        else:
            best_ivs, icpdags = self.optimal_intervention_greedy(cpdag=icpdag, num_interventions=num_interventions-1)
            return [best_iv] + best_ivs, [icpdag] + icpdags

    def fully_orienting_interventions_greedy(self, cpdag=None):
        if cpdag is None:
            cpdag = self.cpdag()
        curr_cpdag = cpdag
        ivs = []
        icpdags = []
        while len(curr_cpdag.edges) != 0:
            iv, icpdag = self.optimal_intervention_greedy(cpdag=curr_cpdag)
            iv = iv[0]
            icpdag = icpdag[0]
            curr_cpdag = icpdag
            ivs.append(iv)
            icpdags.append(icpdag)
        return ivs, icpdags

    def backdoor(self, i, j):
        """
        Returns S satisfying the backdoor criterion if such an S exists, otherwise False.

        S satisfies the backdoor criterion if
        (i) S blocks every path from i to j with an arrow into i
        (ii) no node in S is a descendant of i

        :param i:
        :param j:
        :return:
        """
        raise NotImplementedError
        pass

    def frontdoor(self, i, j):
        """
        Returns S satisfying the frontdoor criterion if such an S exists, otherwise False.

        S satisfies the frontdoor criterion if
        (i) S blocks all directed paths from i to j
        (ii) there are no unblocked backdoor paths from i to S
        (iii) i blocks all backdoor paths from S to j
        :param i:
        :param j:
        :return:
        """
        raise NotImplementedError()

    def dsep(self, i, j, c=None):
        raise NotImplementedError()


if __name__ == '__main__':
    d = DAG(arcs={(1, 2), (1, 3), (3, 4), (2, 4), (3, 5)})
    d.save_gml('test_mine.gml')




