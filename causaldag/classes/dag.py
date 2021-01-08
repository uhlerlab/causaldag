# Author: Chandler Squires
"""Base class for causal DAGs
"""

from collections import defaultdict
import numpy as np
import itertools as itr
from causaldag.utils import core_utils
import operator as op
from causaldag.classes.custom_types import Node, DirectedEdge, NodeSet, warn_untested
from typing import Set, Union, Tuple, Any, Iterable, Dict, FrozenSet, List
import networkx as nx
from networkx.utils import UnionFind
import random
import csv
import ipdb
from scipy.special import comb


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

    def __init__(self, nodes: Set = frozenset(), arcs: Set = frozenset(), dag=None):
        if dag is not None:
            self._nodes = set(dag._nodes)
            self._arcs = set(dag._arcs)
            self._neighbors = defaultdict(set)
            for node, nbrs in dag._neighbors.items():
                self._neighbors[node] = set(nbrs)
            self._parents = defaultdict(set)
            for node, par in dag._parents.items():
                self._parents[node] = set(par)
            self._children = defaultdict(set)
            for node, ch in dag._children.items():
                self._children[node] = set(ch)
        else:
            self._nodes = set(nodes)
            self._arcs = set()
            self._neighbors = defaultdict(set)
            self._parents = defaultdict(set)
            self._children = defaultdict(set)
            # print('before call to add arcs from')
            self.add_arcs_from(arcs, check_acyclic=True)

    def __eq__(self, other):
        if not isinstance(other, DAG):
            return False
        return self._nodes == other._nodes and self._arcs == other._arcs

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

    def copy(self):
        """
        Return a copy of the current DAG.
        """
        # return DAG(nodes=self._nodes, arcs=self._arcs)
        return DAG(dag=self)

    def rename_nodes(self, name_map: Dict):
        """
        Rename the nodes in this graph according to ``name_map``.

        Parameters
        ----------
        name_map:
            A dictionary from the current name of each node to the desired name of each node.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={('a', 'b'), ('b', 'c')})
        >>> g2 = g.rename_nodes({'a': 1, 'b': 2, 'c': 3})
        >>> g2.arcs
        {(1, 2), (2, 3)}
        """
        return DAG(
            nodes={name_map[n] for n in self._nodes},
            arcs={(name_map[i], name_map[j]) for i, j in self._arcs}
        )

    def induced_subgraph(self, nodes: Set[Node]):
        """
        Return the induced subgraph over only ``nodes``

        Parameters
        ----------
        nodes:
            Set of nodes for the induced subgraph.

        Returns
        -------
        DAG:
            Induced subgraph over ``nodes``.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(1, 2), (2, 3), (1, 4)})
        >>> d_induced = d.induced_subgraph({1, 2, 3})
        >>> d_induced.arcs
        {(1, 2), (2, 3)}
        """
        return DAG(nodes, {(i, j) for i, j in self._arcs if i in nodes and j in nodes})

    # === PROPERTIES
    @property
    def nodes(self) -> Set[Node]:
        return set(self._nodes)

    @property
    def nnodes(self) -> int:
        return len(self._nodes)

    @property
    def arcs(self) -> Set[DirectedEdge]:
        return set(self._arcs)

    @property
    def num_arcs(self) -> int:
        return len(self._arcs)

    @property
    def neighbors(self) -> Dict[Node, Set[Node]]:
        return core_utils.defdict2dict(self._neighbors, self._nodes)

    @property
    def parents(self) -> Dict[Node, Set[Node]]:
        return core_utils.defdict2dict(self._parents, self._nodes)

    @property
    def children(self) -> Dict[Node, Set[Node]]:
        return core_utils.defdict2dict(self._children, self._nodes)

    @property
    def skeleton(self) -> Set[FrozenSet]:
        return {frozenset({i, j}) for i, j in self._arcs}

    @property
    def in_degrees(self) -> Dict[Node, int]:
        return {node: len(self._parents[node]) for node in self._nodes}

    @property
    def out_degrees(self) -> Dict[Node, int]:
        return {node: len(self._children[node]) for node in self._nodes}

    @property
    def max_in_degree(self) -> int:
        return max(len(self._parents[node]) for node in self._nodes)

    @property
    def max_out_degree(self) -> int:
        return max(len(self._parents[node]) for node in self._nodes)

    @property
    def sparsity(self) -> float:
        p = len(self._nodes)
        return len(self._arcs) / p / (p - 1) * 2

    # === NODE PROPERTIES
    def parents_of(self, nodes: NodeSet) -> Set[Node]:
        """
        Return all nodes that are parents of the node or set of nodes ``nodes``.

        Parameters
        ----------
        nodes
            A node or set of nodes.

        See Also
        --------
        children_of, neighbors_of, markov_blanket_of

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g.parents_of(2)
        {1}
        >>> g.parents_of({2, 3})
        {1, 2}
        """
        if isinstance(nodes, set):
            return set.union(*(self._parents[n] for n in nodes))
        else:
            return self._parents[nodes].copy()

    def children_of(self, nodes: NodeSet) -> Set[Node]:
        """
        Return all nodes that are children of the node or set of nodes ``nodes``.

        Parameters
        ----------
        nodes
            A node or set of nodes.

        See Also
        --------
        parents_of, neighbors_of, markov_blanket_of

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g.children_of(1)
        {2}
        >>> g.children_of({1, 2})
        {2, 3}
        """
        if isinstance(nodes, set):
            return set.union(*(self._children[n] for n in nodes))
        else:
            return self._children[nodes].copy()

    def neighbors_of(self, nodes: NodeSet) -> Set[Node]:
        """
        Return all nodes that are adjacent to the node or set of nodes ``node``.

        Parameters
        ----------
        nodes
            A node or set of nodes.

        See Also
        --------
        parents_of, children_of, markov_blanket_of

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(0,1), (0,2)})
        >>> g.neighbors_of(0)
        {1, 2}
        >>> g.neighbors_of(2)
        {0}
        """
        if isinstance(nodes, set):
            return set.union(*(self._neighbors[n] for n in nodes))
        else:
            return self._neighbors[nodes].copy()

    def markov_blanket_of(self, node: Node) -> set:
        """
        Return the Markov blanket of ``node``, i.e., the parents of the node, its children, and the parents of its children.

        Parameters
        ----------
        node:
            Node whose Markov blanket to return.

        See Also
        --------
        parents_of, children_of, neighbors_of

        Returns
        -------
        set:
            the Markov blanket of ``node``.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(0, 1), (1, 3), (2, 3), (3, 4})
        >>> g.markov_blanket_of(1)
        {0, 2, 3}
        """
        parents_of_children = set.union(*(self._parents[c] for c in self._children[node])) if self._children[
            node] else set()
        return self._parents[node] | self._children[node] | parents_of_children - {node}

    def is_ancestor_of(self, anc: Node, desc: Node) -> bool:
        """
        Check if ``anc`` is an ancestor of ``desc``

        Return
        ------
        bool
            True if ``anc`` is an ancestor  of ``desc``

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.is_ancestor_of(1, 3)
        True
        >>> g.is_ancestor_of(3, 1)
        False
        """
        return desc in self._children[anc] or desc in self.descendants_of(anc)

    def _add_descendants(self, descendants, node):
        for child in self._children[node]:
            if child not in descendants:
                descendants.add(child)
                self._add_descendants(descendants, child)

    def descendants_of(self, nodes: NodeSet) -> Set[Node]:
        """
        Return the descendants of ``node``.

        Parameters
        ----------
        nodes:
            The node.

        See Also
        --------
        ancestors_of

        Return
        ------
        Set[node]
            Return all nodes j such that there is a directed path from ``node`` to j.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g.descendants_of(1)
        {2, 3}
        """
        descendants = set()
        if not isinstance(nodes, set):
            self._add_descendants(descendants, nodes)
        else:
            return set.union(*(self.descendants_of(node) for node in nodes))
        return descendants

    def _add_ancestors(self, ancestors, node):
        for parent in self._parents[node]:
            if parent not in ancestors:
                ancestors.add(parent)
                self._add_ancestors(ancestors, parent)

    def ancestors_of(self, nodes: Node) -> Set[Node]:
        """
        Return the ancestors of ``nodes``.

        Parameters
        ----------
        nodes:
            The node.

        See Also
        --------
        descendants_of

        Return
        ------
        Set[node]
            Return all nodes j such that there is a directed path from j to ``node``.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g.ancestors_of(3)
        {1, 2, 3}
        """
        ancestors = set()
        if not isinstance(nodes, set):
            self._add_ancestors(ancestors, nodes)
        else:
            return set.union(*(self.ancestors_of(node) for node in nodes))
        return ancestors

    def incident_arcs(self, node: Node) -> Set[DirectedEdge]:
        """
        Return all arcs with ``node`` as either source or target.

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
            Return all arcs i->j such that either i=``node`` of j=``node``.

        Example
        -------
        >>> import causaldag as cd
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

    def incoming_arcs(self, node: Node) -> Set[DirectedEdge]:
        """
        Return all arcs with target ``node``.

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
            Return all arcs of the form i->``node``.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.incoming_arcs(2)
        {(1, 2)}
        """
        incoming_arcs = set()
        for parent in self._parents[node]:
            incoming_arcs.add((parent, node))
        return incoming_arcs

    def outgoing_arcs(self, node: Node) -> Set[DirectedEdge]:
        """
        Return all arcs with source ``node``.

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
            Return all arcs of the form ``node``->j.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.outgoing_arcs(2)
        {(2, 3)}
        """
        outgoing_arcs = set()
        for child in self._children[node]:
            outgoing_arcs.add((node, child))
        return outgoing_arcs

    def outdegree_of(self, node: Node) -> int:
        """
        Return the outdegree of ``node``.

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        indegree_of

        Return
        ------
        int
            The number of children of ``node``.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.outdegree_of(1)
        2
        >>> g.outdegree_of(3)
        0
        """
        return len(self._children[node])

    def indegree_of(self, node: Node) -> int:
        """
        Return the indegree of ``node``.

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        outdegree_of

        Return
        ------
        int
            The number of parents of ``node``.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.indegree_of(1)
        0
        >>> g.indegree_of(2)
        2
        """
        return len(self._parents[node])

    # ==== ORDERS
    def topological_sort(self) -> List[Node]:
        """
        Return a topological sort of the nodes in the graph.

        Returns
        -------
        List[Node]
            A topological sort of the nodes in a graph.

        Examples
        --------
        >>> import causaldag as cd
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

    def is_topological(self, order: list) -> bool:
        """
        Check that ``order`` is a topological order consistent with this DAG, i.e., if ``i``->``j`` in the DAG,
        then ``i`` comes before ``j`` in the order.

        Parameters
        ----------
        order:
            the order to check.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3)})
        >>> g.is_topological([1, 2, 3])
        True
        >>> g.is_topological([1, 3, 2])
        True
        >>> g.is_topological([2, 1, 3])
        False
        """
        node2ix = {node: ix for ix, node in enumerate(order)}
        return all(node2ix[i] < node2ix[j] for i, j in self._arcs)

    def permutation_score(self, order: list) -> int:
        """
        Return the number of "errors" in ``order`` with respect to the DAG, i.e., the number of times that ``i``->``j``
        in the DAG but ``i`` comes *after* ``j`` in ``order``.

        Parameters
        ----------
        order:
            the order to check.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3)})
        >>> g.permutation_score([1, 2, 3])
        0
        >>> g.permutation_score([2, 1, 3])
        1
        >>> g.permutation_score([2, 3, 1])
        2
        """
        node2ix = {node: ix for ix, node in enumerate(order)}
        return sum(node2ix[i] > node2ix[j] for i, j in self._arcs)

    # === GRAPH MODIFICATION
    def add_node(self, node: Node):
        """
        Add ``node`` to the DAG.

        Parameters
        ----------
        node:
            a hashable Python object

        See Also
        --------
        add_nodes_from

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG()
        >>> g.add_node(1)
        >>> g.add_node(2)
        >>> len(g.nodes)
        2
        """
        self._nodes.add(node)

    def add_nodes_from(self, nodes: Iterable):
        """
        Add nodes to the graph from the collection ``nodes``.

        Parameters
        ----------
        nodes:
            collection of nodes to be added.

        See Also
        --------
        add_node

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG({1, 2})
        >>> g.add_nodes_from({'a', 'b'})
        >>> g.add_nodes_from(range(3, 6))
        >>> g.nodes
        {1, 2, 'a', 'b', 3, 4, 5}
        """
        for node in nodes:
            self.add_node(node)

    def remove_node(self, node: Node, ignore_error=False):
        """
        Remove the node ``node`` from the graph.

        Parameters
        ----------
        node:
            node to be removed.
        ignore_error:
            if True, ignore the KeyError raised when node is not in the DAG.

        Examples
        --------
        >>> import causaldag as cd
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
            self._neighbors.pop(node, None)
            self._parents.pop(node, None)
            self._children.pop(node, None)
            self._arcs = {(i, j) for i, j in self._arcs if i != node and j != node}

        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def add_arc(self, i: Node, j: Node, check_acyclic=True):
        """
        Add the arc ``i`` -> ``j`` to the DAG

        Parameters
        ----------
        i:
            source node of the arc
        j:
            target node of the arc
        check_acyclic:
            if True, check that the DAG remains acyclic after adding the edge.

        See Also
        --------
        add_arcs_from

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG({1, 2})
        >>> g.add_arc(1, 2)
        >>> g.arcs
        {(1, 2)}
        """
        self._nodes.add(i)
        self._nodes.add(j)
        self._arcs.add((i, j))

        self._neighbors[i].add(j)
        self._neighbors[j].add(i)

        self._children[i].add(j)
        self._parents[j].add(i)

        if check_acyclic:
            try:
                self._check_acyclic()
            except CycleError as e:
                self.remove_arc(i, j)
                raise e

    def add_arcs_from(self, arcs: Iterable[Tuple], check_acyclic=False):
        """
        Add arcs to the graph from the collection ``arcs``.

        Parameters
        ----------
        arcs:
            collection of arcs to be added.
        check_acyclic:
            if True, check that the DAG remains acyclic after adding the edge.

        See Also
        --------
        add_arcs

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2)})
        >>> g.add_arcs_from({(1, 3), (2, 3)})
        >>> g.arcs
        {(1, 2), (1, 3), (2, 3)}
        """
        if not isinstance(arcs, set):
            arcs = {(i, j) for i, j in arcs}
        if len(arcs) == 0:
            return

        sources, sinks = zip(*arcs)
        self._nodes.update(sources)
        self._nodes.update(sinks)
        self._arcs.update(arcs)
        for i, j in arcs:
            self._neighbors[i].add(j)
            self._neighbors[j].add(i)
            self._children[i].add(j)
            self._parents[j].add(i)

        if check_acyclic:
            try:
                self._check_acyclic()
            except CycleError as e:
                for i, j in arcs:
                    self.remove_arc(i, j)
                raise e

    def remove_arc(self, i: Node, j: Node, ignore_error=False):
        """
        Remove the arc ``i`` -> ``j``.

        Parameters
        ----------
        i:
            source of arc to be removed.
        j:
            target of arc to be removed.
        ignore_error:
            if True, ignore the KeyError raised when arc is not in the DAG.

        Examples
        --------
        >>> import causaldag as cd
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

    def remove_arcs_from(self, arcs: Iterable, ignore_error=False):
        """
        Remove each arc in ``arcs`` from the DAG.

        Parameters
        ----------
        arcs
            The arcs to be removed from the DAG.
        ignore_error:
            if True, ignore the KeyError raised when an arc is not in the DAG.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (2, 3), (3, 4)})
        >>> g.remove_arcs_from({(1, 2), (2, 3)})
        >>> g.arcs
        {(3, 4)}
        """
        for i, j in arcs:
            self.remove_arc(i, j, ignore_error=ignore_error)

    def reverse_arc(self, i: Node, j: Node, ignore_error=False, check_acyclic=False):
        """
        Reverse the arc ``i`` -> ``j`` to ``i`` <- ``j``.

        Parameters
        ----------
        i:
            source of arc to be reversed.
        j:
            target of arc to be reversed.
        ignore_error:
            if True, ignore the KeyError raised when arc is not in the DAG.
        check_acyclic:
            if True, check that the DAG remains acyclic after adding the edge.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2)})
        >>> g.reverse_arc(1, 2)
        >>> g.arcs
        {(2, 1)}
        """
        self.remove_arc(i, j, ignore_error=ignore_error)
        self.add_arc(j, i, check_acyclic=check_acyclic)

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

    # === GRAPH PROPERTIES
    def has_arc(self, source: Node, target: Node) -> bool:
        """
        Check if this DAG has an arc ``source`` -> ``target``.

        Parameters
        ----------
        source:
            Source node of arc.
        target:
            Target node of arc.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(0,1), (0,2)})
        >>> g.has_arc(0, 1)
        True
        >>> g.has_arc(1, 2)
        False
        """
        return (source, target) in self._arcs

    def sources(self) -> Set[Node]:
        """
        Get all nodes in the graph that have no parents.

        Return
        ------
        List[node]
            Nodes in the graph that have no parents.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.sources()
        {1}
        """
        return {node for node in self._nodes if len(self._parents[node]) == 0}

    def sinks(self) -> Set[Node]:
        """
        Get all nodes in the graph that have no children.

        Return
        ------
        List[node]
            Nodes in the graph that have no children.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.sinks()
        {3}
        """
        return {node for node in self._nodes if len(self._children[node]) == 0}

    def reversible_arcs(self) -> Set[DirectedEdge]:
        """
        Get all reversible (aka covered) arcs in the DAG.

        Return
        ------
        Set[arc]
            Return all reversible (aka covered) arcs in the DAG. An arc i -> j is *covered* if the :math:`Pa(j) = Pa(i) \cup {i}`.
            Reversing a reversible arc results in a DAG in the same Markov equivalence class.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.reversible_arcs()
        {(1, 2), (2, 3)}
        """
        reversible_arcs = set()
        for i, j in self._arcs:
            if self._parents[i] == (self._parents[j] - {i}):
                reversible_arcs.add((i, j))
        return reversible_arcs

    def is_reversible(self, i: Node, j: Node) -> bool:
        """
        Check if the arc ``i`` -> ``j`` is reversible (aka covered), i.e., if :math:`pa(i) = pa(j) \setminus \{i\}`

        Parameters
        ----------
        i:
            source of the arc
        j:
            target of the arc

        Returns
        -------
        True if the arc is reversible, otherwise False.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3)})
        >>> g.is_reversible(1, 2)
        True
        >>> g.is_reversible(1, 3)
        False
        """
        return self._parents[i] == self._parents[j] - {i}

    def arcs_in_vstructures(self) -> Set[Tuple]:
        """
        Get all arcs in the graph that participate in a v-structure.

        Return
        ------
        Set[arc]
            Return all arcs in the graph in a v-structure (aka an immorality). A v-structure is formed when i->j<-k but
            there is no arc between i and k. Arcs that participate in a v-structure are identifiable from observational
            data.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 3), (2, 3)})
        >>> g.arcs_in_vstructures()
        {(1, 3), (2, 3))
        """
        return {(i, j) for i, j in self._arcs if self._parents[j] - self._neighbors[i] - {i}}

    def vstructures(self) -> Set[Tuple]:
        """
        Get all v-structures in the graph, i.e., triples of the form (i, k, j) such that ``i``->k<-``j`` and ``i``
        is not adjacent to ``j``.

        Return
        ------
        Set[Tuple]
            Return all triples in the graph in a v-structure (aka an immorality). A v-structure is formed when i->j<-k but
            there is no arc between i and k. Arcs that participate in a v-structure are identifiable from observational
            data.

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 3), (2, 3)})
        >>> g.vstructures()
        {(1, 3, 2)}
        """
        vstructs = set()
        for node in self._nodes:
            for p1, p2 in itr.combinations(self._parents[node], 2):
                if p1 not in self._parents[p2] and p2 not in self._parents[p1]:
                    vstructs.add((p1, node, p2))
        return vstructs

    def triples(self) -> Set[Tuple]:
        """
        Return all triples of the form (``i``, ``j``, ``k``) such that ``i`` and ``k`` are both adjacent to ``j``.

        Returns
        -------
        Set[Tuple]
            Triples in the graph.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 3), (2, 3), (1, 2)})
        >>> g.triples()
        {frozenset({1, 3, 2})}
        """
        t = set()
        for node in self._nodes:
            t |= {frozenset({n1, node, n2}) for n1, n2 in itr.combinations(self._neighbors[node], 2)}
        return t

    def upstream_most(self, s: Set[Node]) -> Set[Node]:
        """
        Return the set of nodes which in ``s`` which have no ancestors in ``s``.

        Parameters
        ----------
        s:
            Set of nodes

        Returns
        -------
        The set of nodes in ``s`` with no ancestors in ``s``.
        """
        return {node for node in s if not self.ancestors_of(node) & s}

    # === COMPARISON
    def shd(self, other) -> int:
        """
        Compute the structural Hamming distance between this DAG and the DAG ``other``.

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
        >>> import causaldag as cd
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
        """
        Compute the structure Hamming distance between the skeleton of this DAG and the skeleton of the graph ``other``.

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
        >>> import causaldag as cd
        >>> g1 = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g2 = cd.DAG(arcs={(2, 1), (2, 3)})
        >>> g1.shd_skeleton(g2)
        0

        >>> g1 = cd.DAG(arcs={(1, 2)})
        >>> g2 = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g1.shd_skeleton(g2)
        1
        """
        return len(self.skeleton.symmetric_difference(other.skeleton))

    def markov_equivalent(self, other, interventions=None) -> bool:
        """
        Check if this DAG is (interventionally) Markov equivalent to the DAG ``other``.

        Parameters
        ----------
        other:
            Another DAG.
        interventions:
            If not None, check whether the two DAGs are interventionally Markov equivalent under the interventions.

        Examples
        --------
        >>> import causaldag as cd
        >>> d1 = cd.DAG(arcs={(0, 1), (1, 2)})
        >>> d2 = cd.DAG(arcs={(2, 1), (1, 0)})
        >>> d3 = cd.DAG(arcs={(0, 1), (2, 1)})
        >>> d4 = cd.DAG(arcs={(1, 0), (1, 2)})
        >>> d1.markov_equivalent(d2)
        True
        >>> d2.markov_equivalent(d1)
        True
        >>> d1.markov_equivalent(d3)
        False
        >>> d1.markov_equivalent(d2, [{2}])
        False
        >>> d1.markov_equivalent(d4, [{2}])
        True
        """
        if interventions is None:
            return self.cpdag() == other.cpdag()
        else:
            return self.interventional_cpdag(interventions, self.cpdag()) == other.interventional_cpdag(interventions,
                                                                                                        other.cpdag())

    def is_imap(self, other) -> bool:
        """
        Check if this DAG is an IMAP of the DAG ``other``, i.e., all d-separation statements in this graph
        are also d-separation statements in ``other``.

        Parameters
        ----------
        other:
            Another DAG.

        See Also
        --------
        is_minimal_imap

        Returns
        -------
        bool
            True if ``other`` is an I-MAP of this DAG, otherwise False.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (3, 2)})
        >>> other = cd.DAG(arcs={(1, 2)})
        >>> g.is_imap(other)
        True
        >>> other = cd.DAG(arcs={(1, 2), (2, 3)})
        >>> g.is_imap(other)
        False
        """
        return all(other.dsep(node, nondesc, parents) for node, nondesc, parents in self.local_markov_statements())

    def is_minimal_imap(self, other, certify=False, check_imap=True) -> Union[bool, Tuple[bool, Any]]:
        """
        Check if this DAG is a minimal IMAP of `other`, i.e., it is an IMAP and no proper subgraph of this DAG
        is an IMAP of other. Deleting the arc i->j retains IMAPness when `i` is d-separated from `j` in `other`
        given the parents of `j` besides `i` in this DAG.

        Parameters
        ----------
        other:
            Another DAG.
        certify:
            If True and this DAG is not an IMAP of other, return a certificate of non-minimality in the form
            of an edge i->j that can be deleted while retaining IMAPness.
        check_imap:
            If True, first check whether this DAG is an IMAP of other, if False, this DAG is assumed to be an IMAP
            of other.

        See Also
        --------
        is_imap

        Returns
        -------
        bool
            True if ``other`` is a minimal I-MAP of this DAG, otherwise False.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (3, 2)})
        >>> other = cd.DAG(arcs={(1, 2)})
        >>> g.is_minimal_imap(other)
        False
        """
        if check_imap and not self.is_imap(other):
            if certify:
                return False, None
            else:
                return False

        certificate = next(((i, j) for i, j in self._arcs if other.dsep(i, j, self._parents[j] - {i})), None)
        if certify:
            return certificate is None, certificate
        else:
            return certificate is None

    def chickering_distance(self, other) -> int:
        """
        Return the total number of edge reversals plus twice the number of edge additions/deletions required
        to turn this DAG into the DAG ``other``.

        Parameters
        ----------
        other:
            the DAG against which to compare the Chickering distance.

        Returns
        -------
        int
            The Chickering distance between this DAG and the DAG ``other``.

        Examples
        --------
        >>> import causaldag as cd
        >>> d1 = cd.DAG(arcs={(0, 1), (1, 2)})
        >>> d2 = cd.DAG(arcs={(0, 1), (2, 1), (3, 1)})
        >>> d1.chickering_distance(d2)
        3
        """
        reversals = self._arcs & {tuple(reversed(arc)) for arc in other._arcs}
        return len(reversals) + 2 * self.shd_skeleton(other)

    def confusion_matrix(self, other, rates_only=False):
        """
        Return the "confusion matrix" associated with estimating the CPDAG of ``other`` instead of the CPDAG of this DAG.

        Parameters
        ----------
        other:
            The DAG against which to compare.
        rates_only:
            if True, the dictionary of results only contains the false positive rate, true positive rate, and precision.

        Returns
        -------
        dict
            Dictionary of results

            * false_positive_arcs:
                the arcs in the CPDAG of ``other`` which are not arcs or edges in the CPDAG of this DAG.
            * false_positive_edges:
                the edges in the CPDAG of ``other`` which are not arcs or edges in the CPDAG of this DAG.
            * false_negative_arcs:
                the arcs in the CPDAG of this graph which are not arcs or edges in the CPDAG of ``other``.
            * true_positive_arcs:
                the arcs in the CPDAG of ``other`` which are arcs in the CPDAG of this DAG.
            * reversed_arcs:
                the arcs in the CPDAG of ``other`` whose reversals are arcs in the CPDAG of this DAG.
            * mistaken_arcs_for_edges:
                the arcs in the CPDAG of ``other`` whose reversals are arcs in the CPDAG of this DAG.
            * false_negative_edges:
                the edges in the CPDAG of this DAG which are not arcs or edges in the CPDAG of ``other``.
            * true_positive_edges:
                the edges in the CPDAG of ``other`` which are edges in the CPDAG of this DAG.
            * mistaken_edges_for_arcs:
                the edges in the CPDAG of ``other`` which are arcs in the CPDAG of this DAG.
            * num_false_positives:
                the total number of: false_positive_arcs, false_positive_edges
            * num_false_negatives:
                the total number of: false_negative_arcs, false_negative_edges, mistaken_arcs_for_edges, and reversed_arcs
            * num_true_positives:
                the total number of: true_positive_arcs, true_positive_edges, and mistaken_edges_for_arcs
            * num_true_negatives:
                the total number of missing arcs/edges in ``other`` which are actually missing in this DAG.
            * fpr:
                the false positive rate, i.e., num_false_positives/(num_false_positives+num_true_negatives). If this DAG
                is fully connected, defaults to 0.
            * tpr:
                the true positive rate, i.e., num_true_positives/(num_true_positives+num_false_negatives). If this DAG
                is empty, defaults to 1.
            * precision:
                the precision, i.e., num_true_positives/(num_true_positives+num_false_positives). If ``other`` is
                empty, defaults to 1.

        Examples
        --------
        >>> import causaldag as cd
        >>> d1 = cd.DAG(arcs={(0, 1), (1, 2)})
        >>> d2 = cd.DAG(arcs={(0, 1), (2, 1)})
        >>> cm = d1.confusion_matrix(d2)
        >>> cm["mistaken_edges_for_arcs"]
        {frozenset({0, 1}), frozenset({1, 2})},
        >>> cm = d2.confusion_matrix(d1)
        >>> cm["mistaken_arcs_for_edges"]
        {(0, 1), (2, 1)}
        """
        self_cpdag = self.cpdag()

        from causaldag.classes.pdag import PDAG
        if isinstance(other, PDAG):
            other_cpdag = other
        else:
            other_cpdag = other.cpdag()

        # HELPER SETS SELF
        self_arcs_as_edges = {frozenset(arc) for arc in self_cpdag._arcs}
        self_edges_as_arcs1 = {(i, j) for i, j in self_cpdag._edges}
        self_edges_as_arcs2 = {(j, i) for i, j in self_edges_as_arcs1}

        # HELPER SETS OTHER
        other_arcs_reversed = {(j, i) for i, j in other_cpdag._arcs}
        other_arcs_as_edges = {frozenset(arc) for arc in other_cpdag._arcs}
        other_edges_as_arcs1 = {(i, j) for i, j in other_cpdag._edges}
        other_edges_as_arcs2 = {(j, i) for i, j in other_edges_as_arcs1}

        # MISSING IN TRUE GRAPH
        false_positive_arcs = other_cpdag._arcs - self_cpdag._arcs - self_edges_as_arcs1 - self_edges_as_arcs2
        false_positive_edges = other_cpdag._edges - self_cpdag._edges - self_arcs_as_edges

        # ARC IN TRUE GRAPH
        false_negative_arcs = self_cpdag._arcs - other_cpdag._arcs - other_edges_as_arcs1 - other_edges_as_arcs2
        true_positive_arcs = self_cpdag._arcs & other_cpdag._arcs
        reversed_arcs = self_cpdag._arcs & other_arcs_reversed
        mistaken_arcs_for_edges = self_cpdag._arcs & (other_edges_as_arcs1 | other_edges_as_arcs2)

        # EDGE IN TRUE GRAPH
        false_negative_edges = self_cpdag._edges - other_cpdag._edges - other_arcs_as_edges
        true_positive_edges = self_cpdag._edges & other_cpdag._edges
        mistaken_edges_for_arcs = self_cpdag._edges & other_arcs_as_edges

        # COMBINED_RESULTS
        num_false_positives = len(false_positive_edges) + len(false_negative_arcs)
        num_false_negatives = len(false_negative_arcs) + len(false_negative_edges) + len(mistaken_arcs_for_edges) + len(
            reversed_arcs)
        num_true_positives = len(true_positive_edges) + len(true_positive_arcs) + len(mistaken_edges_for_arcs)
        num_true_negatives = comb(self.nnodes, 2) - num_false_positives - num_false_negatives - num_true_positives

        # RATES
        num_negatives = comb(self.nnodes, 2) - self.num_arcs
        num_positives = self.num_arcs
        num_returned_positives = (num_true_positives + num_false_positives)
        fpr = num_false_positives / num_negatives if num_negatives != 0 else 0
        tpr = num_true_positives / num_positives if num_positives != 0 else 1
        precision = num_true_positives / num_returned_positives if num_returned_positives != 0 else 1

        if rates_only:
            return dict(
                fpr=fpr,
                tpr=tpr,
                precision=precision
            )

        res = dict(
            false_positive_arcs=false_positive_arcs,
            false_positive_edges=false_positive_edges,
            false_negative_arcs=false_negative_arcs,
            true_positive_arcs=true_positive_arcs,
            reversed_arcs=reversed_arcs,
            mistaken_arcs_for_edges=mistaken_arcs_for_edges,
            false_negative_edges=false_negative_edges,
            true_positive_edges=true_positive_edges,
            mistaken_edges_for_arcs=mistaken_edges_for_arcs,
            num_false_positives=num_false_positives,
            num_false_negatives=num_false_negatives,
            num_true_positives=num_true_positives,
            num_true_negatives=num_true_negatives,
            fpr=fpr,
            tpr=tpr,
            precision=precision
        )

        return res

    def confusion_matrix_skeleton(self, other):
        """
        Return the "confusion matrix" associated with estimating the skeleton of ``other`` instead of the skeleton of
        this DAG.

        Parameters
        ----------
        other:
            The DAG against which to compare.

        Returns
        -------
        dict
            Dictionary of results

            * false_positives:
                the edges in the skeleton of ``other`` which are not in the skeleton of this DAG.
            * false_negatives:
                the edges in the skeleton of this graph which are not in the skeleton of ``other``.
            * true_positives:
                the edges in the skeleton of ``other`` which are acutally in the skeleton of this DAG.
            * num_false_positives:
                the total number of false_positives
            * num_false_negatives:
                the total number of false_negatives
            * num_true_positives:
                the total number of true_positives
            * num_true_negatives:
                the total number of missing edges in the skeleton of ``other`` which are actually missing in this DAG.
            * fpr:
                the false positive rate, i.e., num_false_positives/(num_false_positives+num_true_negatives). If this DAG
                is fully connected, defaults to 0.
            * tpr:
                the true positive rate, i.e., num_true_positives/(num_true_positives+num_false_negatives). If this DAG
                is empty, defaults to 1.
            * precision:
                the precision, i.e., num_true_positives/(num_true_positives+num_false_positives). If ``other`` is
                empty, defaults to 1.

        Examples
        --------
        >>> import causaldag as cd
        >>> d1 = cd.DAG(arcs={(0, 1), (1, 2)})
        >>> d2 = cd.DAG(arcs={(0, 1), (2, 1)})
        >>> cm = d1.confusion_matrix_skeleton(d2)
        >>> cm["tpr"]
        1.0
        >>> d3 = cd.DAG(arcs={(0, 1), (0, 2)})
        >>> cm = d2.confusion_matrix_skeleton(d3)
        >>> cm["true_positives"]
        {frozenset({0, 1})}
        >>> cm["false_positives"]
        {frozenset({0, 2})},
        >>> cm["false_negatives"]
        {frozenset({1, 2})}
        """
        self_skeleton = self.skeleton
        other_skeleton = other.skeleton

        true_positives = self_skeleton & other_skeleton
        false_positives = other_skeleton - self_skeleton
        false_negatives = self_skeleton - other_skeleton

        num_true_positives = len(true_positives)
        num_false_positives = len(false_positives)
        num_false_negatives = len(false_negatives)
        num_true_negatives = comb(self.nnodes, 2) - num_true_positives - num_false_positives - num_false_negatives

        num_positives = len(self_skeleton)
        num_negatives = comb(self.nnodes, 2) - num_positives

        tpr = num_true_positives / num_positives if num_positives != 0 else 1
        fpr = num_false_positives / num_negatives if num_negatives != 0 else 0

        res = dict(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            num_true_positives=num_true_positives,
            num_false_positives=num_false_positives,
            num_true_negatives=num_true_negatives,
            num_false_negatives=num_false_negatives,
            tpr=tpr,
            fpr=fpr
        )

        return res

    # === WRITING TO FILES
    @classmethod
    def from_gml(cls, filename):
        raise NotImplementedError

    @classmethod
    def from_csv(cls, filename):
        raise NotImplementedError

    def save_gml(self, filename):
        """
        TODO
        """
        raise NotImplementedError
        warn_untested()  # TODO: ADD TEST

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

    def to_csv(self, filename):
        """
        TODO
        """
        raise NotImplementedError
        warn_untested()  # TODO: ADD TEST

        with open(filename, 'w', newline='\n') as file:
            writer = csv.writer(file)
            for source, target in self._arcs:
                writer.writerow([source, target])

    # === NUMPY CONVERSION
    @classmethod
    def from_amat(cls, amat: np.ndarray):
        """
        Return a DAG with arcs given by ``amat``, i.e. i->j if ``amat[i,j] != 0``.

        Parameters
        ----------
        amat:
            Numpy matrix representing arcs in the DAG.

        Examples
        --------
        >>> import causaldag as cd
        >>> import numpy as np
        >>> amat = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        >>> d = cd.DAG.from_amat(amat)
        >>> d.arcs
        {(0, 2), (1, 2)}
        """
        nodes = set(range(amat.shape[0]))
        arcs = {(i, j) for i, j in itr.permutations(nodes, 2) if amat[i, j] != 0}
        return DAG(nodes=nodes, arcs=arcs)

    def to_amat(self, node_list=None) -> (np.ndarray, list):
        """
        Return an adjacency matrix for this DAG.

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
        >>> import causaldag as cd
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

    # === NETWORKX CONVERSION
    @classmethod
    def from_nx(cls, nx_graph: nx.DiGraph):
        """
        Convert a networkx DiGraph into a DAG.

        Parameters
        ----------
        nx_graph:
            networkx DiGraph

        Returns
        -------
        DAG:
            The graph as a DAG object.

        Examples
        --------
        >>> import causaldag as cd
        >>> import networkx as nx
        >>> g = nx.DiGraph()
        >>> g.add_edges_from([(0, 1)])
        >>> d = cd.DAG.from_nx(g)
        >>> d.arcs
        {(0, 1)}
        """
        if not isinstance(nx_graph, nx.DiGraph):
            raise ValueError("Must be a DiGraph")
        return DAG(nodes=set(nx_graph.nodes), arcs=set(nx_graph.edges))

    def to_nx(self) -> nx.DiGraph:
        """
        Convert DAG to a networkx DiGraph.

        Returns
        -------
        networkx.DiGraph:
            The graph as a networkx.DiGraph object.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(0, 1)})
        >>> g = d.to_nx()
        >>> g.edges
        OutEdgeView([(0, 1)])
        """
        g = nx.DiGraph()
        g.add_nodes_from(self._nodes)
        g.add_edges_from(self._arcs)
        return g

    # === PANDAS CONVERSION
    @classmethod
    def from_dataframe(cls, df):
        """
        Create a DAG from a dataframe, where the indices and columns are node names and a nonzero entry indicates
        the presence of an edge.

        Parameters
        ----------
        df:
            The pandas dataframe.

        Returns
        -------
        DAG:
            The graph as a DAG object.

        Examples
        --------
        >>> import causaldag as cd
        >>> import numpy as np
        >>> import pandas as pd
        >>> amat = np.array([[0, 1], [0, 0]])
        >>> df = pd.DataFrame(amat, index=["a", "b"], columns=["a", "b"])
        >>> d = cd.DAG.from_dataframe(df)
        >>> d.arcs
        {('a', 'b')}
        """
        warn_untested()  # TODO: ADD TEST

        g = DAG(nodes=set(df.index) | set(df.columns))
        for (i, j), val in np.ndenumerate(df.values):
            if val != 0:
                g.add_arc(df.index[i], df.columns[j])
        return g

    def to_dataframe(self, node_list=None):
        """
        Turn this DAG into a dataframe, where the indices and columns are node names and a nonzero entry indicates
        the presence of an edge.

        Parameters
        ----------
        node_list:
            Order to use when creating the dataframe. If None, uses a sorted order.

        Returns
        -------
        pandas.DataFrame:
            The graph as a DataFrame.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(0, 1)})
        >>> d.to_dataframe()
           0  1
        0  0  1
        1  0  0
        >>> d.to_dataframe(node_list=[1, 0])
           1  0
        1  0  0
        0  1  0
        """
        warn_untested()  # TODO: ADD TEST

        if not node_list:
            node_list = sorted(self._nodes)
        node2ix = {node: i for i, node in enumerate(node_list)}

        shape = (len(self._nodes), len(self._nodes))
        amat = np.zeros(shape, dtype=int)
        for source, target in self._arcs:
            amat[node2ix[source], node2ix[target]] = 1

        from pandas import DataFrame
        return DataFrame(amat, index=node_list, columns=node_list)

    # === SCIPY CONVERSION
    def to_sparse(self):
        raise NotImplementedError

    # === SEPARATIONS
    def dsep(self, A: Union[Set[Node], Node], B: Union[Set[Node], Node], C: Union[Set[Node], Node] = set(),
             verbose=False, certify=False) -> bool:
        """
        Check if ``A`` and ``B`` are d-separated given ``C``, using the Bayes ball algorithm.

        Parameters
        ----------
        A:
            First set of nodes.
        B:
            Second set of nodes.
        C:
            Separating set of nodes.
        verbose:
            If True, print moves of the algorithm.

        See Also
        --------
        dsep_from_given

        Return
        ------
        is_dsep

        Example
        -------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (3, 2)})
        >>> g.dsep(1, 3)
        True
        >>> g.dsep(1, 3, 2)
        False
        """
        warn_untested()  # TODO: ADD TEST

        # type coercion
        A = core_utils.to_set(A)
        B = core_utils.to_set(B)
        C = core_utils.to_set(C)

        # shade ancestors of C
        shaded_nodes = set(C)
        for node in C:
            self._add_ancestors(shaded_nodes, node)

        visited = set()
        # marks for which direction the path is traveling through the node
        _c = '_c'  # child
        _p = '_p'  # parent

        schedule = {(node, _c) for node in A}
        while schedule:
            if verbose:
                print('Current schedule:', schedule)

            node, _dir = schedule.pop()
            if node in B and not certify: return False
            if node in B and certify: return False, node
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

    def dsep_from_given(self, A, C: NodeSet = frozenset()) -> Set[Node]:
        """
        Find all nodes d-separated from ``A`` given ``C``.

        Uses algorithm in Geiger, D., Verma, T., & Pearl, J. (1990).
        Identifying independence in Bayesian networks. Networks, 20(5), 507-534.

        Parameters
        ----------
        A:
            set of nodes.
        C:
            set of conditioned nodes.

        Returns
        -------
        set
            Nodes which are d-separated from ``A`` given ``C``.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(0, 1), (1, 2), (2, 3), (3, 4)})
        >>> d.dsep_from_given(0, 1)
        {2, 3, 4]
        """
        warn_untested()  # TODO: ADD TEST

        A = core_utils.to_set(A)
        C = core_utils.to_set(C)

        determined = set()
        descendants = set()

        for c in C:
            determined.add(c)
            descendants.add(c)
            self._add_ancestors(descendants, c)

        reachable = set()
        i_links = set()
        labeled_links = set()

        for a in A:
            i_links.add((None, a))
            reachable.add(a)

        while True:
            i_p_1_links = set()
            # Find all unlabled links v->w adjacent to at least one link u->v labeled i, such that (u->v,v->w) is a legal pair.
            for link in i_links:
                u, v = link
                for w in self._neighbors[v]:
                    if not u == w and (v, w) not in labeled_links:
                        if v in self._children[u] and v in self._children[w]:  # Is collider?
                            if v in descendants:
                                i_p_1_links.add((v, w))
                                reachable.add(w)
                        else:  # Not collider
                            if v not in determined:
                                i_p_1_links.add((v, w))
                                reachable.add(w)

            if len(i_p_1_links) == 0:
                break

            labeled_links = labeled_links.union(i_links)
            i_links = i_p_1_links

        return self._nodes.difference(A).difference(C).difference(reachable)

    def is_invariant(self, A, intervened_nodes, cond_set=set(), verbose=False) -> bool:
        """
        Check if the distribution of ``A`` given cond_set is invariant to an intervention on intervened_nodes.

        :math:`f^\emptyset(A|C) = f^I(A|C)` if the "intervention node" I with intervened_nodes as its children
        is d-separated from A given C. Equivalently, the :math:`f^\emptyset(A|C) \neq f^I(A|C)` if:

        - there is an active path to an intervened node that ends in an arrowhead, and that intervened node
            or one of its descendants is conditioned on.
        - there is an active path to an intervened node that ends in a tail, and that intervened node
            is not conditioned on.

        Parameters
        ----------
        A:
            Set of nodes.
        intervened_nodes:
            Nodes on which an intervention has occurred.
        cond_set:
            Conditioning set for the tested distribution.
        verbose:
            If True, print moves of the algorithm.
        """
        warn_untested()  # TODO: ADD TEST

        # type coercion
        A = core_utils.to_set(A)
        I = core_utils.to_set(intervened_nodes)
        C = core_utils.to_set(cond_set)

        # shade ancestors of C
        shaded_nodes = set(C)
        for node in C:
            self._add_ancestors(shaded_nodes, node)

        visited = set()
        # marks for which direction the path is traveling through the node
        _c = '_c'  # child
        _p = '_p'  # parent

        schedule = {(node, _c) for node in A}
        while schedule:
            if verbose:
                print('Current schedule:', schedule)

            node, _dir = schedule.pop()
            if node in I and _dir == _p and node in shaded_nodes: return False
            if node in I and _dir == _c and node not in C: return False
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

    def local_markov_statements(self) -> Set[Tuple[Any, FrozenSet, FrozenSet]]:
        """
        Return the local Markov statements of this DAG, i.e., those of the form ``i`` independent nondescendants(i) given
        the parents of ``i``.

        Returns
        -------
        set
            The set of tuples of the form (``i``, ``A``, ``C``) representing the local Markov statements of the DAG
            via (``i`` independent of ``A`` given ``C``).

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (3, 2)})
        >>> g.local_markov_statements()
        {(1, frozenset({3}), frozenset()), (2, frozenset(), frozenset({1, 3})), (3, frozenset({1}), frozenset())}
        """
        statements = set()
        for node in self._nodes:
            parents = self._parents[node]
            nondescendants = self._nodes - {node} - self.descendants_of(node) - parents
            statements.add((node, frozenset(nondescendants), frozenset(parents)))
        return statements

    # === CONVERSION TO OTHER GRAPHS
    def moral_graph(self):
        """
        Return the (undirected) moral graph of this DAG, i.e., the graph with the parents of all nodes made adjacent.

        Returns
        -------
        UndirectedGraph:
            Moral graph of this DAG.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(1, 3), (2, 3)})
        >>> ug = d.moral_graph()
        >>> ug.edges
        {frozenset({1, 3}), frozenset({2, 3}), frozenset({1, 2})}
        """
        warn_untested()  # TODO: ADD TEST

        from causaldag import UndirectedGraph
        edges = {(i, j) for i, j in self._arcs} | {(p1, p2) for p1, node, p2 in self.vstructures()}
        return UndirectedGraph(self._nodes, edges)

    def marginal_mag(self, latent_nodes, relabel=None, new=True):
        """
        Return the maximal ancestral graph (MAG) that results from marginalizing out ``latent_nodes``.

        Parameters
        ----------
        latent_nodes:
            nodes to marginalize over.
        relabel:
            if relabel='default', relabel the nodes to have labels 1,2,...,(#nodes).
        new:
            TODO - pick whether to use new or old implementation.

        Returns
        -------
        m:
            cd.AncestralGraph, the MAG resulting from marginalizing out `latent_nodes`.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(1, 3), (1, 2)})
        >>> mag = d.marginal_mag(latent_nodes={1})
        >>> mag
        Directed edges: set(), Bidirected edges: {frozenset({2, 3})}, Undirected edges: set()
        >>> mag = d.marginal_mag(latent_nodes={1}, relabel="default")
        Directed edges: set(), Bidirected edges: {frozenset({0, 1})}, Undirected edges: set()
        """
        warn_untested()  # TODO: ADD TEST

        from .ancestral_graph import AncestralGraph

        if not new:
            latent_nodes = core_utils.to_set(latent_nodes)

            new_nodes = self._nodes - latent_nodes
            directed = set()
            bidirected = set()
            for i, j in itr.combinations(self._nodes - latent_nodes, r=2):
                adjacent = all(not self.dsep(i, j, S) for S in core_utils.powerset(self._nodes - {i, j} - latent_nodes))
                if adjacent:
                    if self.is_ancestor_of(i, j):
                        directed.add((i, j))
                    elif self.is_ancestor_of(j, i):
                        directed.add((j, i))
                    else:
                        bidirected.add((i, j))

            if relabel is not None:
                t = self.topological_sort()
                t_new = [node for node in t if node not in latent_nodes]
                node2new_label = dict(map(reversed, enumerate(t_new)))
                new_nodes = {node2new_label[node] for node in new_nodes}
                directed = {(node2new_label[i], node2new_label[j]) for i, j in directed}
                bidirected = {(node2new_label[i], node2new_label[j]) for i, j in bidirected}

            return AncestralGraph(nodes=new_nodes, directed=directed, bidirected=bidirected)

        else:
            # ag = AncestralGraph(nodes=self._nodes, directed=self._arcs)
            # curr_directed = ag.directed
            # curr_bidirected = ag.bidirected
            #
            # while True:
            #     for node in latent_nodes:
            #         parents = ag._parents[node]
            #         children = ag._children[node]
            #         spouses = ag._spouses[node]
            #         for j, i in itr.product(parents, children):
            #             ag._add_directed(j, i, ignore_error=True)
            #         for i, j in itr.combinations(children, 2):
            #             ag._add_bidirected(i, j, ignore_error=True)
            #         for i, j in itr.product(children, spouses):
            #             ag._add_bidirected(i, j, ignore_error=True)
            #
            #     last_directed = curr_directed
            #     last_bidirected = curr_bidirected
            #     curr_directed = ag.directed
            #     curr_bidirected = ag.bidirected
            #     if curr_directed == last_directed and curr_bidirected == last_bidirected:
            #         break
            # for node in latent_nodes:
            #     ag.remove_node(node, ignore_error=True)

            ag = AncestralGraph(nodes=self._nodes, directed=self._arcs)
            ancestor_dict = ag.ancestor_dict()
            for i, j in itr.combinations(self._nodes - latent_nodes, 2):
                S = (ancestor_dict[i] | ancestor_dict[j]) - {i, j} - latent_nodes
                if not ag.has_any_edge(i, j) and not ag.msep(i, j, S):
                    if i in ancestor_dict[j]:
                        ag._add_directed(i, j)
                    elif j in ancestor_dict[i]:
                        ag._add_directed(j, i)
                    else:
                        ag._add_bidirected(i, j)
            for node in latent_nodes:
                ag.remove_node(node, ignore_error=True)

            if relabel is not None:
                if relabel == 'default':
                    relabel = {node: ix for ix, node in enumerate(sorted(self._nodes - set(latent_nodes)))}
                new_nodes = {relabel[node] for node in self._nodes - set(latent_nodes)}
                directed = {(relabel[i], relabel[j]) for i, j in ag.directed}
                bidirected = {(relabel[i], relabel[j]) for i, j in ag.bidirected}
                return AncestralGraph(new_nodes, directed=directed, bidirected=bidirected)

            return ag

    def cpdag(self):
        """
        Return the completed partially directed acyclic graph (CPDAG, aka essential graph) that represents the
        Markov equivalence class of this DAG.

        Return
        ------
        causaldag.PDAG:
            CPDAG representing the MEC of this DAG.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (2, 4), (3, 4)})
        >>> cpdag = g.cpdag()
        >>> cpdag.edges
        {frozenset({1, 2})}
        >>> cpdag.arcs
        {(2, 4), (3, 4)}
        """
        from causaldag import PDAG
        pdag = PDAG(nodes=self._nodes, arcs=self._arcs, known_arcs=self.arcs_in_vstructures())
        pdag.remove_unprotected_orientations()
        return pdag

    def cpdag_new(self, new=False):
        from causaldag import PDAG
        vstruct = self.arcs_in_vstructures()
        pdag = PDAG(nodes=self._nodes, arcs=vstruct, edges=self._arcs - vstruct)
        if new:
            pdag.to_complete_pdag_new()
        else:
            pdag.to_complete_pdag()
        return pdag

    def interventional_cpdag(self, interventions: List[set], cpdag=None):
        """
        Return the interventional essential graph (aka CPDAG) associated with this DAG.

        Parameters
        ----------
        interventions:
            A list of the intervention targets.
        cpdag:
            The original (non-interventional) CPDAG of the graph. Faster when provided.

        Return
        ------
        causaldag.PDAG:
            Interventional CPDAG representing the I-MEC of this DAG.

        Examples
        --------
        >>> import causaldag as cd
        >>> g = cd.DAG(arcs={(1, 2), (2, 4), (3, 4)})
        >>> cpdag = g.cpdag()
        >>> icpdag = g.interventional_cpdag([{1}], cpdag=cpdag)
        >>> icpdag.arcs
        {(1, 2), (2, 4), (3, 4)}
        """
        warn_untested()  # TODO: ADD TEST

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

    # === CHICKERING SEQUENCE
    def _is_resolved_sink(self, other, node, res_sinks):
        no_children = not (self._children[node] - res_sinks)
        no_children_other = not (other._children[node] - res_sinks)
        same_parents = self._parents[node] == other._parents[node]
        return no_children and no_children_other and same_parents

    def resolved_sinks(self, other) -> set:
        """
        Return the nodes in this graph which are "resolved sinks" with respect to the graph ``other``.

        A "resolved sink" is a node which has the same parents in both graphs, and no children which are
        not themselves resolved sinks.

        Parameters
        ----------
        other
            TODO

        Examples
        --------
        >>> import causaldag as cd
        >>> d1 = cd.DAG(arcs={(1, 0), (1, 2), (2, 0)})
        >>> d2 = cd.DAG(arcs={(2, 0), (2, 1), (1, 0)})
        >>> res_sinks = d1.resolved_sinks(d2)
        {0}
        """
        warn_untested()  # TODO: ADD TEST

        res_sinks = set()
        while True:
            new_resolved = {node for node in self._nodes - res_sinks if self._is_resolved_sink(other, node, res_sinks)}
            res_sinks.update(new_resolved)
            if not new_resolved:
                break

        return res_sinks

    def chickering_sequence(self, imap, verbose=False):
        """
        Return a *Chickering sequence* from this DAG to an I-MAP ``imap``.

        A Chickering sequence from DAG ``D1`` to a DAG ``D2`` is a sequence of DAGs starting at ``D1`` and ending at
        ``D2``, with consecutive DAGs differing by a single edge reversal or edge deletion, such that each DAG is an
        IMAP of ``D1``.

        See Chickering, David Maxwell. "Optimal structure identification with greedy search." (2002) for more details.

        Parameters
        ----------
        imap: DAG
            The I-MAP of this DAG at which the Chickering sequence will end.

        Examples
        --------
        >>> import causaldag as cd
        >>> d1 = cd.DAG(arcs={(0, 1), (1, 2)})
        >>> d2 = cd.DAG(arcs={(2, 0), (2, 1), (1, 0)})
        >>> sequence, moves = d1.chickering_sequence(d2)
        >>> sequence[1].arcs
        {(1, 0), (1, 2)}
        >>> sequence[2].arcs
        {(1, 0), (1, 2), (2, 0)}
        """
        warn_untested()  # TODO: ADD TEST

        curr_graph = self

        ch_seq = []
        moves = []
        last_sink = None
        while curr_graph != imap:
            ch_seq.append(curr_graph)
            curr_graph, last_sink, move = curr_graph.apply_edge_operation(imap, seed_sink=last_sink, verbose=verbose)
            moves.append(move)

        ch_seq.append(imap)
        return ch_seq, moves

    def apply_edge_operation(self, imap, seed_sink=None, verbose=False):
        """
        Identify an edge operation (covered edge reversal or edge addition) which decreases the Chickering distance
        from this DAG to ``imap``.

        See Chickering, David Maxwell. "Optimal structure identification with greedy search." (2002), Fig. 2 for
        more details.

        Parameters
        ----------
        imap:
            The target I-MAP.
        seed_sink:
            If the algorithm reaches step 3, pick this node (if it is indeed a valid sink).
        verbose:
            If ``True``, print out the steps of the algorithm.

        Returns
        -------
        (DAG, Node, int)
            * The updated DAG
            * The node picked for the operation
            * The type of the edge operation (corresponding to the line of the algorithm in the above paper)
        """
        warn_untested()  # TODO: ADD TEST

        new_graph = self.copy()

        # STEP 2: REMOVE RESOLVED SINKS
        resolved_sinks = self.resolved_sinks(imap)
        self_subgraph = self.induced_subgraph(self._nodes - resolved_sinks)
        imap_subgraph = imap.induced_subgraph(imap._nodes - resolved_sinks)

        # STEP 3: PICK A SINK IN THE IMAP
        imap_sinks = imap_subgraph.sinks()
        sink = random.choice(list(imap_sinks)) if seed_sink is None or seed_sink not in imap_sinks else seed_sink
        if verbose: print(f"Step 3: Picked {sink}")

        # STEP 4: ADD A PARENT IF Y IS A SINK IN G
        if sink in self_subgraph.sinks():
            x = random.choice(list(imap_subgraph._parents[sink] - self_subgraph._parents[sink]))
            new_graph.add_arc(x, sink)
            if verbose: print(f"Step 4: Added {x}->{sink}")
            return new_graph, sink, 4

        # STEP 5: PICK A SPECIFIC CHILD OF Y IN G
        d = list(imap_subgraph.upstream_most(self_subgraph.descendants_of(sink)))[0]
        valid_children = self_subgraph.upstream_most(self_subgraph._children[sink]) & (
                self_subgraph.ancestors_of(d) | {d})
        z = random.choice(list(valid_children))
        if verbose: print(f"Step 5: Picked z={z}")

        # STEP 6
        if self_subgraph.is_reversible(sink, z):
            new_graph.reverse_arc(sink, z)
            if verbose: print(f"Step 6: Reversing {sink}->{z}")
            return new_graph, sink, 6

        # STEP 7
        par_z = self_subgraph._parents[z] - self_subgraph._parents[sink] - {sink}
        if par_z:
            x = random.choice(list(par_z))
            if verbose: print(f"Step 7: Picked x={x}")
            new_graph.add_arc(x, sink)
            if verbose: print(f"Step 7: Adding {x}->{sink}")
            return new_graph, sink, 7

        # STEP 8
        par_sink = self_subgraph._parents[sink] - self_subgraph._parents[z]
        x = random.choice(list(par_sink))
        if verbose: print(f"Step 8: Picked x={x}")
        new_graph.add_arc(x, z)
        if verbose: print(f"Step 8: Adding {x}->{z}")
        return new_graph, sink, 8

    # === DIRECTED CLIQUE TREES
    def directed_clique_tree(self, verbose=False):
        """
        Return the directed clique tree associated with this DAG.

        See the following for the definition of the directed clique tree:
        Squires, Chandler, et al. "Active Structure Learning of Causal DAGs via Directed Clique Tree." (2020)

        Parameters
        ----------
        verbose
            if True, print out the steps taken to compute the directed clique tree.

        Returns
        -------
        networkx.MultiDiGraph
            The directed clique tree of this DAG.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(0, 1), (1, 2), (1, 3), (2, 3)})
        >>> dct = d.directed_clique_tree()
        >>> dct.nodes
        NodeView((frozenset({1, 2, 3}), frozenset({0, 1})))
        >>> dct.edges
        OutMultiEdgeView([(frozenset({0, 1}), frozenset({1, 2, 3}), 0)])
        """
        warn_untested()  # TODO: ADD TEST

        cliques = nx.chordal_graph_cliques(self.to_nx().to_undirected())
        ct = nx.MultiDiGraph()
        ct.add_nodes_from(cliques)
        edges = {(c1, c2): c1 & c2 for c1, c2 in itr.combinations(cliques, 2) if c1 & c2}
        subtrees = UnionFind()
        bidirected_components = UnionFind()
        for c1, c2 in sorted(edges, key=lambda e: len(edges[e]), reverse=True):
            if verbose: print(f"Considering edge {c1}-{c2}")
            if subtrees[c1] != subtrees[c2]:
                shared = c1 & c2
                all_into_c1 = all((s, c) in self._arcs for s, c in itr.product(shared, c1 - shared))
                all_into_c2 = all((s, c) in self._arcs for s, c in itr.product(shared, c2 - shared))
                if all_into_c1 and all_into_c2:
                    c1_parent = bidirected_components[c1]
                    c2_parent = bidirected_components[c2]
                    b1 = [c for c, parent in bidirected_components.parents.items() if parent == c1_parent]
                    b2 = [c for c, parent in bidirected_components.parents.items() if parent == c2_parent]
                    b1_source = any(set(ct.predecessors(c)) - set(ct.successors(c)) for c in b1)
                    b2_source = any(set(ct.predecessors(c)) - set(ct.successors(c)) for c in b2)
                    if not (b1_source and b2_source):
                        if verbose: print(f"Adding edge {c1}<->{c2}")
                        subtrees.union(c1, c2)
                        bidirected_components.union(c1, c2)
                        ct.add_edge(c1, c2)
                        ct.add_edge(c2, c1)
                else:
                    c1, c2 = (c1, c2) if all_into_c2 else (c2, c1)
                    c2_parent = bidirected_components[c2]
                    bidirected_component = [
                        c for c, parent in bidirected_components.parents.items()
                        if parent == c2_parent
                    ]
                    has_source = any(
                        set(ct.predecessors(c)) - set(ct.successors(c))
                        for c in bidirected_component
                    )
                    if not has_source:
                        if verbose: print(f"{c1}->{c2}")
                        ct.add_edge(c1, c2)
                        subtrees.union(c1, c2)

        labels = {(c1, c2, 0): c1 & c2 for c1, c2 in ct.edges()}
        nx.set_edge_attributes(ct, labels, name='label')

        return ct

    def contracted_directed_clique_tree(self):
        """
        Return the contracted directed clique tree associated with this DAG.

        See the following for the definition of the contracted directed clique tree:
        Squires, Chandler, et al. "Active Structure Learning of Causal DAGs via Directed Clique Tree." (2020)

        Returns
        -------
        networkx.MultiDiGraph
            The directed clique tree of this DAG.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(0, 1), (1, 2), (1, 3), (1, 4), (3, 2), (3, 4)})
        >>> cdct = d.contracted_directed_clique_tree()
        >>> cdct.nodes
        NodeView((frozenset({frozenset({1, 2, 3}), frozenset({1, 3, 4})}), frozenset({frozenset({0, 1})})))
        >>> cdct.edges
        OutEdgeView([(frozenset({frozenset({0, 1})}), frozenset({frozenset({1, 2, 3}), frozenset({1, 3, 4})}))])
        """
        warn_untested()  # TODO: ADD TEST

        dct = self.directed_clique_tree()

        # find bidirected connected components
        all_edges = set(dct.edges())
        bidirected_graph = nx.Graph()
        bidirected_graph.add_nodes_from(dct.nodes())
        bidirected_graph.add_edges_from({(c1, c2) for c1, c2 in all_edges if (c2, c1) in all_edges})
        components = [frozenset(component) for component in nx.connected_components(bidirected_graph)]
        clique2component = {clique: component for component in components for clique in component}

        # contract bidirected connected components
        g = nx.DiGraph()
        g.add_nodes_from(components)
        g.add_edges_from({
            (clique2component[c1], clique2component[c2]) for c1, c2 in all_edges
            if clique2component[c1] != clique2component[c2]
        })

        return g

    def residuals(self):
        """
        Return the residuals associated with this DAG.

        See the following for the definition of residuals:
        Squires, Chandler, et al. "Active Structure Learning of Causal DAGs via Directed Clique Tree." (2020)

        Returns
        -------
        networkx.MultiDiGraph
            The directed clique tree of this DAG.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(0, 1), (1, 2), (1, 3), (1, 4), (3, 2), (3, 4)})
        >>> residuals = d.residuals()
        >>> residuals
        [frozenset({2, 3, 4}), frozenset({0, 1})]
        """
        warn_untested()  # TODO: ADD TEST

        sdct = self.contracted_directed_clique_tree()
        sdct_nodes = list(sdct.nodes)
        sdct_components = [frozenset.union(*component) for component in sdct_nodes]
        sdct_parents = [list(sdct.predecessors(component)) for component in sdct_nodes]
        sdct_parents = [frozenset.union(*p[0]) if p else set() for p in sdct_parents]
        return [component - parent for component, parent in zip(sdct_components, sdct_parents)]

    def residual_essential_graph(self):
        """
        Return the residual essential graph associated with this DAG.

        See the following for the definition of the residual essential graph:
        Squires, Chandler, et al. "Active Structure Learning of Causal DAGs via Directed Clique Tree." (2020)

        Returns
        -------
        networkx.MultiDiGraph
            The directed clique tree of this DAG.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(0, 1), (1, 2), (1, 3), (1, 4), (3, 2), (3, 4)})
        >>> r_eg = d.residual_essential_graph()
        >>> r_eg.arcs
        {(1, 2), (1, 3), (1, 4)}
        """
        warn_untested()  # TODO: ADD TEST

        from causaldag import PDAG

        sdct = self.contracted_directed_clique_tree()
        sdct_nodes = list(sdct.nodes)
        sdct_components = [frozenset.union(*component) for component in sdct_nodes]
        sdct_parents = [list(sdct.predecessors(component)) for component in sdct_nodes]
        sdct_parents = [frozenset.union(*p[0]) if p else set() for p in sdct_parents]
        sdct_residuals = [component - parent for component, parent in zip(sdct_components, sdct_parents)]
        arcs = {
            (p, r) for parent, residual, component in zip(sdct_parents, sdct_residuals, sdct_components)

            for p, r in itr.product(parent & component, residual)
        }
        g = PDAG(nodes=self._nodes, arcs=arcs & self._arcs, edges=self._arcs - arcs)
        return g

    # === INTERVENTION DESIGN
    def optimal_fully_orienting_single_node_interventions(self, cpdag=None, new=False, verbose=False) -> Set[Node]:
        """
        Find the smallest set of interventions which fully orients the CPDAG into this DAG.

        Parameters
        ----------
        cpdag
            the starting CPDAG containing known orientations. If None, compute and use the observational essential graph.
        new:
            TODO: remove after checking that directed clique tree method works.
        verbose:
            TODO: describe.

        Returns
        -------
        interventions
            A minimum-size set of interventions which fully orients the DAG.

        Examples
        --------
        >>> import causaldag as cd
        >>> import itertools as itr
        >>> d = cd.DAG(arcs=set(itr.combinations(range(5), 2)))
        >>> ivs = d.optimal_fully_orienting_single_node_interventions()
        >>> ivs
        {1, 3}
        """
        if new:
            sdct = self.contracted_directed_clique_tree()
            top_sort = nx.topological_sort(sdct)

            intervened_nodes = set()
            for component in top_sort:
                parent = list(sdct.predecessors(component))
                parent_nodes = frozenset.union(*parent[0]) if len(parent) != 0 else set()
                if verbose: print(f"orienting component {component}, parent={parent}")
                component_intervened_nodes = self._verification_optimal_helper(component, parent_nodes, verbose=verbose)
                if verbose: print(f"intervened: {component_intervened_nodes}")
                intervened_nodes.update(component_intervened_nodes)
            return intervened_nodes
        else:
            cpdag = self.cpdag() if cpdag is None else cpdag
            node2oriented = {
                node: self.interventional_cpdag([{node}], cpdag=cpdag).arcs
                for node in self._nodes - cpdag.dominated_nodes
            }
            for ss in core_utils.powerset(self._nodes - cpdag.dominated_nodes, r_min=1):
                oriented = set.union(*(node2oriented[node] for node in ss))
                if len(oriented) == len(cpdag.edges) + len(cpdag.arcs):
                    return ss

    def greedy_optimal_single_node_intervention(self, cpdag=None, num_interventions=1):
        """
        Greedily pick ``num_interventions`` single node interventions based on how many edges they orient.

        By submodularity, this will orient at least (1 - 1/e) as many edges as the optimal intervention set
        of size ``num_interventions``.

        Parameters
        ----------
        cpdag:
            the starting CPDAG containing known orientations. If None, use the observational essential graph.
        num_interventions:
            the number of single-node interventions used. Default is 1.

        Return
        ------
        (interventions, cpdags)
            The selected interventions and the associated cpdags that they induce.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(0, 1), (1, 2), (0, 2)})
        >>> ivs, icpdags = d.greedy_optimal_single_node_intervention()
        >>> ivs
        [1]
        >>> icpdags[0].arcs
        {(0, 1), (0, 2), (1, 2)}
        """
        warn_untested()  # TODO: ADD TEST

        if cpdag is None:
            cpdag = self.cpdag()
        if len(cpdag.edges) == 0:
            return [None] * num_interventions, [cpdag] * num_interventions

        nodes2icpdags = {
            node: self.interventional_cpdag([{node}], cpdag=cpdag)
            for node in self._nodes - cpdag.dominated_nodes
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
            best_ivs, icpdags = self.greedy_optimal_single_node_intervention(cpdag=icpdag,
                                                                             num_interventions=num_interventions - 1)
            return [best_iv] + best_ivs, [icpdag] + icpdags

    def greedy_optimal_fully_orienting_interventions(self, cpdag=None):
        """
        Find a set of interventions which fully orients a CPDAG into this DAG, using greedy selection of the
        interventions. By submodularity, the number of interventions is a (1 + ln K) multiplicative approximation
        to the true optimal number of interventions, where K is the number of undirected edges in the CPDAG.

        Parameters
        ----------
        cpdag
            the starting CPDAG containing known orientations. If None, use the observational essential graph.

        Returns
        -------
        (interventions, cpdags)
            The selected interventions and the associated cpdags that they induce.

        Examples
        --------
        >>> import causaldag as cd
        >>> d = cd.DAG(arcs={(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)})
        >>> ivs, icpdags = d.greedy_optimal_fully_orienting_interventions()
        >>> ivs
        [1, 2]
        >>> icpdags[0].edges
        {frozenset({2, 3})}
        >>> icpdags[1].edges
        set()
        """
        warn_untested()  # TODO: ADD TEST

        if cpdag is None: cpdag = self.cpdag()
        curr_cpdag = cpdag
        ivs = []
        icpdags = []
        while len(curr_cpdag.edges) != 0:
            iv, icpdag = self.greedy_optimal_single_node_intervention(cpdag=curr_cpdag)
            iv = iv[0]
            icpdag = icpdag[0]
            curr_cpdag = icpdag
            ivs.append(iv)
            icpdags.append(icpdag)
        return ivs, icpdags

    def _verification_optimal_helper(self, component, parent_component, verbose=False) -> set:
        # for a clique, select every other node
        if len(component) == 1:
            if verbose: print('component is clique')
            residual = list(component)[0] - parent_component
            if verbose: print(f"residual: {residual}")
            sorted_nodes = self.induced_subgraph(residual).topological_sort()
            return set(sorted_nodes[1::2])
        else:
            sorted_nodes = self.induced_subgraph(frozenset.union(*component)).topological_sort()

            # determine common head
            intersections = [c1 & c2 for c1, c2 in itr.combinations(component, 2)]
            common_head = frozenset.union(*intersections) - parent_component
            max_intersection = max(intersections, key=len)
            # if max_intersection != frozenset.union(*intersections):
            #     raise RuntimeError
            sorted_common_head = [node for node in sorted_nodes if node in common_head]
            if verbose: print(f'component contains multiple cliques, common head = {sorted_common_head}')

            # determine heads and tails
            heads = [clique & common_head for clique in component]
            tails = [clique - common_head - parent_component for clique in component]
            sorted_heads = [[node for node in sorted_nodes if node in head] for head in heads]
            sorted_tails = [[node for node in sorted_nodes if node in tail] for tail in tails]

            # add nodes from tails
            intervened_nodes = set()
            for head, tail in zip(sorted_heads, sorted_tails):
                if verbose: print(f"tail={tail}, head={head}")
                intervened_nodes.update(tail[-2::-2])
                if len(tail) % 2 == 1 and len(head) > 0:
                    intervened_nodes.add(head[-1])

            # add remaining nodes from common head
            counter = 0
            for node in sorted_common_head:
                if node in intervened_nodes:
                    counter = 0
                else:
                    counter += 1
                    if counter == 2:
                        intervened_nodes.add(node)
                        counter = 0

            return intervened_nodes
            # TODO: test!

    # === ADJUSTMENT SETS
    def backdoor(self, i, j):
        """
        Return a set of nodes S satisfying the backdoor criterion if such an S exists, otherwise False.

        S satisfies the backdoor criterion if
        (i) S blocks every path from i to j with an arrow into i
        (ii) no node in S is a descendant of i


        """
        raise NotImplementedError
        pass

    def frontdoor(self, i, j):
        """
        Return a set of nodes S satisfying the frontdoor criterion if such an S exists, otherwise False.

        S satisfies the frontdoor criterion if
        (i) S blocks all directed paths from i to j
        (ii) there are no unblocked backdoor paths from i to S
        (iii) i blocks all backdoor paths from S to j

        """
        raise NotImplementedError()


if __name__ == '__main__':
    d = DAG(arcs={(1, 2), (1, 3), (3, 4), (2, 4), (3, 5)})
    d.save_gml('test_mine.gml')
