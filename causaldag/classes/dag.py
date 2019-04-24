# Author: Chandler Squires
"""Base class for causal DAGs
"""

from collections import defaultdict
import numpy as np
import itertools as itr
from causaldag.utils import core_utils
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
    """
    Base class for causal DAGs.
    """
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
        """Return a DAG with arcs given by amat
        """
        nodes = set(range(amat.shape[0]))
        arcs = set()
        d = DAG(nodes=nodes)
        for (i, j), val in np.ndenumerate(amat):
            if val != 0:
                d.add_arc(i, j)
        return d

    def copy(self):
        """Return a copy of the current DAG


        """
        return DAG(nodes=self._nodes, arcs=self._arcs)

    @property
    def nodes(self):
        return set(self._nodes)

    @property
    def nnodes(self):
        return len(self._nodes)

    @property
    def arcs(self):
        return set(self._arcs)

    @property
    def num_arcs(self):
        return len(self._arcs)

    @property
    def neighbors(self):
        return core_utils.defdict2dict(self._neighbors, self._nodes)

    @property
    def parents(self):
        return core_utils.defdict2dict(self._parents, self._nodes)

    @property
    def children(self):
        return core_utils.defdict2dict(self._children, self._nodes)

    @property
    def skeleton(self):
        return {tuple(sorted((i, j))) for i, j in self._arcs}

    @property
    def in_degrees(self):
        return {node: len(self._parents[node]) for node in self._nodes}

    @property
    def out_degrees(self):
        return {node: len(self._children[node]) for node in self._nodes}

    @property
    def max_in_degree(self):
        return max(len(self._parents[node]) for node in self._nodes)

    @property
    def max_out_degree(self):
        return max(len(self._parents[node]) for node in self._nodes)

    def parents_of(self, node):
        return self._parents[node].copy()

    def children_of(self, node):
        return self._children[node].copy()

    def neighbors_of(self, node):
        return self._neighbors[node].copy()

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

    def __repr__(self):
        return str(self)

    # === MUTATORS
    def add_node(self, node):
        """Add a node to the DAG

        Parameters
        ----------
        node:
            a hashable Python object

        See Also
        --------
        add_nodes_from

        Examples
        --------
        >>> g = cd.DAG()
        >>> g.add_node(1)
        >>> g.add_node(2)
        >>> len(g.nodes)
        2
        """
        self._nodes.add(node)

    def add_arc(self, i, j):
        """Add an arc to the DAG

        Parameters
        ----------
        i:
            source node of the arc
        j:
            target node of the arc

        See Also
        --------
        add_arcs_from

        Examples
        --------
        >>> g = cd.DAG({1, 2})
        >>> g.add_arc(1, 2)
        >>> g.arcs
        {(1, 2)}
        """
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
        """Return a topological sort of the nodes in the graph

        Returns
        -------
        List[Node]
            Return a topological sort of the nodes in a graph.

        Examples
        --------
        >>> g = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g.topological_sort
        [1, 2, 3]
        """
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
        """Add nodes to the graph from a collection

        Parameters
        ----------
        nodes:
            collection of nodes to be added

        See Also
        --------
        add_node

        Examples
        --------
        >>> g = cd.DAG({1, 2})
        >>> g.add_nodes_from({'a', 'b'})
        >>> g.add_nodes_from(range(3, 6))
        >>> g.nodes
        {1, 2, 'a', 'b', 3, 4, 5}
        """
        for node in nodes:
            self.add_node(node)

    def add_arcs_from(self, arcs):
        """Add arcs to the graph from a collection

        Parameters
        ----------
        arcs:
            collection of arcs to be added

        See Also
        --------
        add_arcs

        Examples
        --------
        >>> g = cd.DAG(arcs={(1, 2)})
        >>> g.add_arcs_from({(1, 3), (2, 3)})
        >>> g.arcs
        {(1, 2), (1, 3), (2, 3)}
        """
        for i, j in arcs:
            self._add_arc(i, j)
        try:
            self._check_acyclic()
        except CycleError as e:
            for i, j in arcs:
                self.remove_arc(i, j)
            raise e

    def reverse_arc(self, i, j, ignore_error=False):
        """Reverse the arc i->j to i<-j

        Parameters
        ----------
        i:
            source of arc to be reversed
        j:
            target of arc to be reversed
        ignore_error:
            if True, ignore the KeyError raised when arc is not in the DAG

        Examples
        --------
        >>> g = cd.DAG(arcs={(1, 2)})
        >>> g.reverse_arc(1, 2)
        >>> g.arcs
        {(2, 1)}
        """
        try:
            self._arcs.remove((i, j))
            self._parents[j].remove(i)
            self._children[i].remove(j)

            self.add_arc(j, i)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_arc(self, i, j, ignore_error=False):
        """Remove the arc i->j

        Parameters
        ----------
        i:
            source of arc to be removed
        j:
            target of arc to be removed
        ignore_error:
            if True, ignore the KeyError raised when arc is not in the DAG

        Examples
        --------
        >>> g = cd.DAG(arcs={(1, 2)})
        >>> g.remove_arc(1, 2)
        >>> g.arcs
        set()
        """
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
        """Remove a node from the graph

        Parameters
        ----------
        node:
            node to be removed
        ignore_error:
            if True, ignore the KeyError raised when node is not in the DAG

        Examples
        --------
        >>> g = cd.DAG(arcs={(1, 2)})
        >>> g.remove_node(2)
        >>> g.nodes
        {1}
        """
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
    def sources(self):
        """Get all nodes in the graph that have no parents

        Return
        ------
        List[node]
            Nodes in the graph that have no parents

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.sources()
        {1)
        """
        return {node for node in self._nodes if len(self._parents[node]) == 0}

    def sinks(self):
        """Get all nodes in the graph that have no children

        Return
        ------
        List[node]
            Nodes in the graph that have no children

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.sinks()
        {3)
        """
        return {node for node in self._nodes if len(self._children[node]) == 0}

    def reversible_arcs(self):
        """Get all reversible (aka covered) arcs in the DAG.

        Return
        ------
        List[arc]
            Return all reversible (aka covered) arcs in the DAG. An arc i -> j is *covered* if the :math:`Pa(j) = Pa(i) \cup {i}`.
            Reversing a reversible arc results in a DAG in the same Markov equivalence class.

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.reversible_arcs()
        {(1, 2), (2, 3))
        """
        reversible_arcs = set()
        for i, j in self._arcs:
            if self._parents[i] == (self._parents[j] - {i}):
                reversible_arcs.add((i, j))
        return reversible_arcs

    def vstructs(self):
        """Get all arcs in the graph that participate in a v-structure.

        Return
        ------
        List[arc]
            Return all arcs in the graph in a v-structure (aka an immorality). A v-structure is formed when i->j<-k but
            there is no arc between i and k. Arcs that participate in a v-structure are identifiable from observational
            data.

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 3), (2, 3)})
        >>> g.vstructs()
        {(1, 3), (2, 3))
        """
        vstructs = set()
        for node in self._nodes:
            for p1, p2 in itr.combinations(self._parents[node], 2):
                if p1 not in self._parents[p2] and p2 not in self._parents[p1]:
                    vstructs.add((p1, node))
                    vstructs.add((p2, node))
        return vstructs

    # === COMPARISON
    def shd(self, other) -> int:
        """Compute the structural Hamming distance between this DAG and another graph

        Parameters
        ----------
        other:
            the DAG to which the SHD will be computed.

        Return
        ------
        int
            The structural Hamming distance between :math:`G_1` and :math:`G_2` is the minimum number of arc additions,
            deletions, and reversals required to transform :math:`G_1` into :math:`G_2` (and vice versa).

        Example
        -------
        >>> g1 = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g2 = cd.DAG(arcs={(2, 1), (2, 3)})
        >>> g1.shd(g2)
        1
        """
        if isinstance(other, DAG):
            self_arcs_reversed = {(j, i) for i, j in self._arcs}
            other_arcs_reversed = {(j, i) for i, j in other._arcs}

            additions = other._arcs - self._arcs - self_arcs_reversed
            deletions = self._arcs - other._arcs - other_arcs_reversed
            reversals = self.arcs & other_arcs_reversed
            return len(additions) + len(deletions) + len(reversals)

    def shd_skeleton(self, other) -> int:
        """Compute the structure Hamming distance between the skeleton of this DAG and the skeleton of another graph

        Parameters
        ----------
        other:
            the DAG to which the SHD of the skeleton will be computed.

        Return
        ------
        int
            The structural Hamming distance between :math:`G_1` and :math:`G_2` is the minimum number of arc additions,
            deletions, and reversals required to transform :math:`G_1` into :math:`G_2` (and vice versa).

        Example
        -------
        >>> g1 = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g2 = cd.DAG(arcs={(2, 1), (2, 3)})
        >>> g1.shd_skeleton(g2)
        0

        >>> g1 = cd.DAG(arcs={(1, 2)})
        >>> g2 = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g1.shd_skeleton(g2)
        1
        """
        if isinstance(other, DAG):
            return len(self.skeleton.symmetric_difference(other.skeleton))

    def markov_equivalent(self, other, interventions=None) -> bool:
        """Check if this DAG is (interventionally) Markov equivalent to some other DAG
        """
        if interventions is None:
            return self.cpdag() == other.cpdag()
        else:
            return self.interventional_cpdag(interventions, self.cpdag()) == other.interventional_cpdag(interventions, other.cpdag())

    # === CONVENIENCE
    def _add_downstream(self, downstream, node):
        for child in self._children[node]:
            if child not in downstream:
                downstream.add(child)
                self._add_downstream(downstream, child)

    def downstream(self, node):
        """Return the nodes downstream of node

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        upstream

        Return
        ------
        Set[node]
            Return all nodes j such that there is a directed path from node to j.

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g.downstream(1)
        {2, 3}
        """
        downstream = set()
        self._add_downstream(downstream, node)
        return downstream

    def _add_upstream(self, upstream, node):
        for parent in self._parents[node]:
            if parent not in upstream:
                upstream.add(parent)
                self._add_upstream(upstream, parent)

    def upstream(self, node):
        """Return the nodes upstream of node

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        downstream

        Return
        ------
        Set[node]
            Return all nodes j such that there is a directed path from j to node.

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g.upstream(3)
        {1, 2, 3g}
        """
        upstream = set()
        self._add_upstream(upstream, node)
        return upstream

    def incident_arcs(self, node):
        """Return all arcs adjacent to node

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        incoming_arcs, outgoing_arcs

        Return
        ------
        Set[arc]
            Return all arcs i->j such that either i=node of j=node.

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.incident_arcs(2)
        {(1, 2), (2, 3)}
        """
        incident_arcs = set()
        for child in self._children[node]:
            incident_arcs.add((node, child))
        for parent in self._parents[node]:
            incident_arcs.add((parent, node))
        return incident_arcs

    def incoming_arcs(self, node):
        """Return all arcs with target node

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        incident_arcs, outgoing_arcs

        Return
        ------
        Set[arc]
            Return all arcs i->node.

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.incoming_arcs(2)
        {(1, 2)}
        """
        incoming_arcs = set()
        for parent in self._parents[node]:
            incoming_arcs.add((parent, node))
        return incoming_arcs

    def outgoing_arcs(self, node):
        """Return all arcs with source node

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        incident_arcs, incoming_arcs

        Return
        ------
        Set[arc]
            Return all arcs node->j.

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.outgoing_arcs(2)
        {(2, 3)}
        """
        outgoing_arcs = set()
        for child in self._children[node]:
            outgoing_arcs.add((node, child))
        return outgoing_arcs

    def outdegree(self, node):
        """Return the outdegree of node

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        indegree

        Return
        ------
        int
            Return the number of children of node.

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.outdegree(1)
        2
        >>> g.outdegree(3)
        0
        """
        return len(self._children[node])

    def indegree(self, node):
        """Return the indegree of node

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        outdegree

        Return
        ------
        int
            Return the number of parents of node.

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.indegree(1)
        0
        >>> g.indegree(2)
        2
        """
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

    def to_dataframe(self, node_list=None):
        if not node_list:
            node_list = sorted(self._nodes)
        node2ix = {node: i for i, node in enumerate(node_list)}

        shape = (len(self._nodes), len(self._nodes))
        amat = np.zeros(shape, dtype=int)
        for source, target in self._arcs:
            amat[node2ix[source], node2ix[target]] = 1

        from pandas import DataFrame
        return DataFrame(amat, index=node_list, columns=node_list)

    @classmethod
    def from_dataframe(cls, df):
        g = DAG(nodes=set(df.index)|set(df.columns))
        for (i, j), val in np.ndenumerate(df.values):
            if val != 0:
                g.add_arc(df.index[i], df.columns[j])
        return g

    def to_amat(self, node_list=None) -> (np.ndarray, list):
        """Return an adjacency matrix for DAG

        Parameters
        ----------
        node_list:
            List indexing the rows/columns of the matrix.

        See Also
        --------
        from_amat

        Return
        ------
        (amat, node_list)

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.to_amat()[0]
        array([[0, 1, 1],
               [0, 0, 1],
               [0, 0, 0]])
        >>> g.to_amat()[1]
        [1, 2, 3]
        """
        if not node_list:
            node_list = sorted(self._nodes)
        node2ix = {node: i for i, node in enumerate(node_list)}

        shape = (len(self._nodes), len(self._nodes))
        amat = np.zeros(shape, dtype=int)

        for source, target in self._arcs:
            amat[node2ix[source], node2ix[target]] = 1

        return amat, node_list

    # === optimal interventions
    def cpdag(self):
        """Return the completed partially directed acyclic graph (CPDAG, aka essential graph) that represents the
        Markov equivalence class of this DAG
        """
        from causaldag import PDAG
        pdag = PDAG(nodes=self._nodes, arcs=self._arcs, known_arcs=self.vstructs())
        pdag.remove_unprotected_orientations()
        return pdag

    def interventional_cpdag(self, interventions, cpdag=None):
        """Return the interventional essential graph (aka CPDAG) associated with this DAG."""
        from causaldag import PDAG

        if cpdag is None:
            raise ValueError('Need the CPDAG')
            # dag_cut = self.copy()
            # known_arcs = set()
            # for node in intervened_nodes:
            #     for i, j in dag_cut.incoming_arcs(node):
            #         dag_cut.remove_arc(i, j)
            #         known_arcs.update(self.outgoing_arcs(node))
            # known_arcs.update(dag_cut.vstructs())
            # pdag = PDAG(dag_cut._nodes, dag_cut._arcs, known_arcs=known_arcs)
        else:
            cut_edges = set()
            for iv_nodes in interventions:
                cut_edges.update({(i, j) for i, j in self._arcs if len({i, j} & set(iv_nodes)) == 1})
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
            node: self.interventional_cpdag([{node}], cpdag=cpdag)
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
        """Return a set of nodes S satisfying the backdoor criterion if such an S exists, otherwise False.

        S satisfies the backdoor criterion if
        (i) S blocks every path from i to j with an arrow into i
        (ii) no node in S is a descendant of i


        """
        raise NotImplementedError
        pass

    def frontdoor(self, i, j):
        """Return a set of nodes S satisfying the frontdoor criterion if such an S exists, otherwise False.

        S satisfies the frontdoor criterion if
        (i) S blocks all directed paths from i to j
        (ii) there are no unblocked backdoor paths from i to S
        (iii) i blocks all backdoor paths from S to j

        """
        raise NotImplementedError()

    def dsep(self, A, B, C=set(), verbose=False):
        """
        Check if A and B are d-separated given C, using the Bayes ball algorithm.

        Parameters
        ----------
        A:
            First set of nodes.
        B:
            Second set of nodes.
        C:
            Separating set of nodes.

        See Also
        --------
        dsep_from_given

        Return
        ------
        is_dsep

        Example
        -------
        >>> g = cd.DAG(arcs={(1, 2), (3, 2)})
        >>> g.dsep(1, 3)
        True
        >>> g.dsep(1, 3, 2)
        False
        """
        # type coercion
        A = core_utils.to_set(A)
        B = core_utils.to_set(B)
        C = core_utils.to_set(C)

        # shade ancestors of C
        shaded_nodes = set(C)
        for node in C:
            self._add_upstream(shaded_nodes, node)

        visited = set()
        # marks for which direction the path is traveling through the node
        _c = '_c'  # child
        _p = '_p'  # parent

        schedule = {(node, _c) for node in A}
        while schedule:
            if verbose:
                print('Current schedule:', schedule)

            node, _dir = schedule.pop()
            if node in B: return False
            if (node, _dir) in visited: continue
            visited.add((node, _dir))

            if verbose:
                print('Going through node', node, 'in direction', _dir)

            # if coming from child, won't encounter v-structure
            if _dir == _c and node not in C:
                schedule.update({(parent, _c) for parent in self._parents[node]})
                schedule.update({(child, _p) for child in self._children[node]})

            if _dir == _p:
                # if coming from parent and see shaded node, can go through v-structure
                if node in shaded_nodes:
                    schedule.update({(parent, _c) for parent in self._parents[node]})

                # if coming from parent and see unconditioned node, can go through children
                if node not in C:
                    schedule.update({(child, _p) for child in self._children[node]})

        return True

    def dsep_from_given(self, A, C=set()):
        """
        Find all nodes d-separated from A given C .

        Uses algorithm in Geiger, D., Verma, T., & Pearl, J. (1990).
        Identifying independence in Bayesian networks. Networks, 20(5), 507-534.
        """
        A = core_utils.to_set(A)
        C = core_utils.to_set(C)

        determined = set()
        descendants = set()

        for c in C:
            determined.add(c)
            descendants.add(c)
            self._add_upstream(descendants, c)
            
        reachable = set()
        i_links = set()
        labeled_links = set()

        for a in A:
            i_links.add((None, a)) 
            reachable.add(a)
       
        while True:
            i_p_1_links = set()
            #Find all unlabled links v->w adjacent to at least one link u->v labeled i, such that (u->v,v->w) is a legal pair.
            for link in i_links:
                u, v = link
                for w in self._neighbors[v]:
                    if not u == w and (v, w) not in labeled_links:
                        if v in self._children[u] and v in self._children[w]:  #Is collider?
                            if v in descendants:
                                i_p_1_links.add((v,w))
                                reachable.add(w)
                        else:  #Not collider
                            if v not in determined:
                                i_p_1_links.add((v,w))
                                reachable.add(w)

            if len(i_p_1_links) == 0:
                break

            labeled_links = labeled_links.union(i_links)
            i_links = i_p_1_links

        return self._nodes.difference(A).difference(C).difference(reachable)



if __name__ == '__main__':
    d = DAG(arcs={(1, 2), (1, 3), (3, 4), (2, 4), (3, 5)})
    d.save_gml('test_mine.gml')
    



