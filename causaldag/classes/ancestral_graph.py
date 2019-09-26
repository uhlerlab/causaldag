from collections import defaultdict
from causaldag.utils import core_utils
import itertools as itr
import numpy as np
import random


class CycleError(Exception):
    def __init__(self):
        super().__init__()
        # def __init__(self, source, target):
        #     self.source = source
        #     self.target = target
        #     message = '%s -> %s will cause a cycle' % (source, target)
        #     super().__init__(message)


class SpouseError(Exception):
    def __init__(self):
        super().__init__()
        # def __init__(self, ancestor, desc):
        #     self.ancestor = ancestor
        #     self.desc = desc
        #     message = '%s <-> %s cannot be added since %s is an ancestor of %s' % (ancestor, desc, ancestor, desc)
        #     super().__init__(message)


class AdjacentError(Exception):
    def __init__(self, node1, node2, arrow_type):
        self.node1 = node1
        self.node2 = node2
        self.arrow_type = arrow_type
        message = '%s %s %s cannot be added since %s and %s are already adjacent' % (
        node1, arrow_type, node2, node1, node2)
        super().__init__(message)


class NeighborError(Exception):
    def __init__(self, node, neighbors=None, parents=None, spouses=None):
        self.node = node
        self.neighbors = neighbors
        self.parents = parents
        self.spouses = spouses
        if self.neighbors:
            message = 'The node %s has neighbors %s. Nodes cannot have neighbors and parents/spouses.' % (
            node, ','.join(map(str, neighbors)))
        elif self.parents:
            message = 'The node %s has parents %s. Nodes cannot have neighbors and parents/spouses.' % (
            node, ','.join(map(str, parents)))
        elif self.spouses:
            message = 'The node %s has spouses %s. Nodes cannot have neighbors and parents/spouses.' % (
            node, ','.join(map(str, spouses)))
        super().__init__(message)


def path2str(path):
    return '->'.join(map(str, path))


class AncestralGraph:
    """
    Base class for ancestral graphs, used to represent causal models with latent variables.
    """

    def __init__(self, nodes=set(), directed=set(), bidirected=set(), undirected=set()):
        self._nodes = nodes.copy()
        self._directed = set()
        self._bidirected = set()
        self._undirected = set()

        self._neighbors = defaultdict(set)
        self._spouses = defaultdict(set)
        self._parents = defaultdict(set)
        self._children = defaultdict(set)
        self._adjacent = defaultdict(set)

        for i, j in directed:
            self._add_directed(i, j)
        for i, j in bidirected:
            self._add_bidirected(i, j)
        for i, j in undirected:
            self._add_undirected(i, j)

    def __eq__(self, other):
        if not isinstance(other, AncestralGraph):
            return False

        same_nodes = self._nodes == other._nodes
        same_directed = self._directed == other._directed
        same_bidirected = self._bidirected == other._bidirected
        same_undirected = self._undirected == other._undirected
        return same_nodes and same_directed and same_bidirected and same_undirected

    def copy(self):
        """
        Return a copy of the current ancestral graph.
        """
        return AncestralGraph(self.nodes, self.directed, self.bidirected, self.undirected)

    def induced_subgraph(self, nodes):
        new_directed = {(i, j) for i, j in self._directed if i in nodes and j in nodes}
        new_bidirected = {(i, j) for i, j in self._bidirected if i in nodes and j in nodes}
        new_undirected = {(i, j) for i, j in self._undirected if i in nodes and j in nodes}

        return AncestralGraph(nodes, directed=new_directed, bidirected=new_bidirected, undirected=new_undirected)

    def __str__(self):
        return 'Directed edges: %s, Bidirected edges: %s, Undirected edges: %s' % (self._directed, self._bidirected, self._undirected)

    def __repr__(self):
        return str(self)

    # === MUTATORS
    def add_node(self, node):
        """Add a node to the ancestral graph.

        Parameters
        ----------
        node:
            a hashable Python object

        See Also
        --------
        add_nodes_from

        Examples
        --------
        >>> g = cd.AncestralGraph()
        >>> g.add_node(1)
        >>> g.add_node(2)
        >>> len(g.nodes)
        2
        """
        self._nodes.add(node)

    def add_nodes_from(self, nodes):
        """Add a node to the ancestral graph.

        Parameters
        ----------
        nodes:
            an iterable of hashable Python objects

        See Also
        --------
        add_node

        Examples
        --------
        >>> g = cd.AncestralGraph()
        >>> g.add_nodes_from({1, 2})
        >>> len(g.nodes)
        2
        """
        for node in nodes:
            self._nodes.add(node)

    def _check_ancestral(self):
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
                raise CycleError
        for spouse in self._spouses[node]:
            if curr_path_visited[spouse]:
                raise SpouseError
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

    def add_directed(self, i, j):
        self._add_directed(i, j)
        try:
            self._check_ancestral()
        except CycleError as e:
            self.remove_directed(i, j)
            raise e

    def add_bidirected(self, i, j):
        self._add_bidirected(i, j)
        try:
            self._add_bidirected(i, j)
        except CycleError as e:
            self.remove_bidirected(i, j)
            raise e

    def add_undirected(self, i, j):
        self._add_undirected(i, j)

    def _add_directed(self, i, j, ignore_error=False):
        if self.has_directed(i, j):
            return

        # === CHECK REMAINS ANCESTRAL
        if self._neighbors[j]:
            raise NeighborError(j, self._neighbors[j])

        # === CHECK i AND j NOT ALREADY ADJACENT
        if i in self._adjacent[j]:
            if ignore_error:
                if self.has_directed(j, i):
                    self.remove_directed(j, i)
                elif self.has_bidirected(i, j):
                    self.remove_bidirected(i, j)
                else:
                    self.remove_undirected(i, j)
            else:
                raise AdjacentError(i, j, '->')

        self._nodes.add(i)
        self._nodes.add(j)
        self._directed.add((i, j))
        self._parents[j].add(i)
        self._children[i].add(j)

        self._adjacent[i].add(j)
        self._adjacent[j].add(i)

    def _add_bidirected(self, i, j, ignore_error=False):
        if self.has_bidirected(i, j):
            return

        # === CHECK REMAINS ANCESTRAL
        if self._neighbors[i]:
            raise NeighborError(i, neighbors=self._neighbors[i])
        if self._neighbors[j]:
            raise NeighborError(j, neighbors=self._neighbors[j])

        # === CHECK i AND j NOT ALREADY ADJACENT
        if i in self._adjacent[j]:
            if ignore_error:
                if self.has_directed(i, j):
                    self.remove_directed(i, j)
                elif self.has_directed(j, i):
                    self.remove_directed(j, i)
                else:
                    self.remove_undirected(i, j)
            else:
                raise AdjacentError(i, j, '<->')

        self._nodes.add(i)
        self._nodes.add(j)
        self._bidirected.add(tuple(sorted((i, j))))
        self._spouses[j].add(i)
        self._spouses[i].add(j)

        self._adjacent[i].add(j)
        self._adjacent[j].add(i)

    def _add_undirected(self, i, j, ignore_error=False):
        if self.has_undirected(i, j):
            return

        # === CHECK REMAINS ANCESTRAL
        if self._parents[i]:
            raise NeighborError(i, parents=self._parents[i])
        if self._spouses[i]:
            raise NeighborError(i, spouses=self._spouses[i])
        if self._parents[j]:
            raise NeighborError(j, parents=self._parents[j])
        if self._spouses[j]:
            raise NeighborError(j, spouses=self._spouses[j])

        # === CHECK i AND j NOT ALREADY ADJACENT
        if i in self._adjacent[j]:
            if ignore_error:
                if self.has_directed(i, j):
                    self.remove_directed(i, j)
                elif self.has_directed(j, i):
                    self.remove_directed(j, i)
                else:
                    self.remove_bidirected(i, j)
            else:
                raise AdjacentError(i, j, '-')

        self._nodes.add(i)
        self._nodes.add(j)
        self._undirected.add(tuple(sorted((i, j))))
        self._neighbors[j].add(i)
        self._neighbors[i].add(j)

        self._adjacent[i].add(j)
        self._adjacent[j].add(i)

    def remove_node(self, node, ignore_error=False):
        try:
            self._nodes.remove(node)
            for parent in self._parents[node]:
                self._children[parent].remove(node)
                self._adjacent[parent].remove(node)
                self._directed.remove((parent, node))
            for child in self._children[node]:
                self._parents[child].remove(node)
                self._adjacent[child].remove(node)
                self._directed.remove((node, child))
            for spouse in self._spouses[node]:
                self._spouses[spouse].remove(node)
                self._adjacent[spouse].remove(node)
                self._bidirected.remove(tuple(sorted((spouse, node))))
            for nbr in self._neighbors[node]:
                self._neighbors[nbr].remove(node)
                self._adjacent[nbr].remove(node)
                self._undirected.remove(tuple(sorted((nbr, node))))

            del self._children[node]
            del self._parents[node]
            del self._spouses[node]
            del self._neighbors[node]
            del self._adjacent[node]
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_directed(self, i, j, ignore_error=False):
        try:
            self._directed.remove((i, j))
            self._children[i].remove(j)
            self._parents[j].remove(i)
            self._adjacent[i].remove(j)
            self._adjacent[j].remove(i)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_bidirected(self, i, j, ignore_error=False):
        try:
            self._bidirected.remove(tuple(sorted((i, j))))
            self._spouses[i].remove(j)
            self._spouses[j].remove(i)
            self._adjacent[i].remove(j)
            self._adjacent[j].remove(i)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_undirected(self, i, j, ignore_error=False):
        try:
            self._undirected.remove(tuple(sorted((i, j))))
            self._neighbors[i].remove(j)
            self._neighbors[j].remove(i)
            self._adjacent[i].remove(j)
            self._adjacent[j].remove(i)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_edge(self, i, j, ignore_error=False):
        if self.has_bidirected(i, j):
            self.remove_bidirected(i, j)
        elif self.has_directed(i, j):
            self.remove_directed(i, j)
        elif self.has_directed(j, i):
            self.remove_directed(j, i)
        elif self.has_undirected(i, j):
            self.remove_undirected(i, j)
        elif not ignore_error:
            raise KeyError

    def remove_edges(self, edges):
        for i, j in edges:
            self.remove_edge(i, j)

    # === PROPERTIES
    @property
    def nodes(self):
        return self._nodes.copy()

    @property
    def nnodes(self):
        return len(self._nodes)

    @property
    def directed(self):
        return self._directed.copy()

    @property
    def num_directed(self):
        return len(self._directed)

    @property
    def bidirected(self):
        return self._bidirected.copy()

    @property
    def num_bidirected(self):
        return len(self._bidirected)

    @property
    def undirected(self):
        return self._undirected.copy()

    @property
    def num_undirected(self):
        return len(self._undirected)

    @property
    def num_edges(self):
        return self.num_directed + self.num_bidirected + self.num_undirected

    @property
    def skeleton(self):
        return {frozenset({i, j}) for i, j in self._bidirected | self._undirected | self._directed}

    def children_of(self, i):
        return self._children[i].copy()

    def parents_of(self, i):
        return self._parents[i].copy()

    def spouses_of(self, i):
        return self._spouses[i].copy()

    def neighbors_of(self, i):
        return self._neighbors[i].copy()

    def _add_ancestors(self, ancestors, node, exclude_arcs=set()):
        for parent in self._parents[node]:
            if parent not in ancestors and (parent, node) not in exclude_arcs:
                ancestors.add(parent)
                self._add_ancestors(ancestors, parent, exclude_arcs=exclude_arcs)

    def _add_descendants(self, descendants, node, exclude_arcs=set()):
        for child in self._children[node]:
            if child not in descendants and (child, node) not in exclude_arcs:
                descendants.add(child)
                self._add_descendants(descendants, child, exclude_arcs=exclude_arcs)

    def ancestors_of(self, nodes, exclude_arcs=set()):
        """Return the nodes upstream of node

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        descendants_of

        Return
        ------
        Set[node]
            Return all nodes j such that there is a directed path from j to node.

        Example
        -------
        """
        ancestors = set()
        if not isinstance(nodes, set):
            self._add_ancestors(ancestors, nodes, exclude_arcs=exclude_arcs)
        else:
            return set.union(*(self.ancestors_of(node) for node in nodes))
        return ancestors

    def ancestor_dict(self):
        """Return a dictionary from each node to its ancestors

        See Also
        --------
        ancestors_of

        Return
        ------
        Dict[node,Set]
            Mapping node to ancestors

        Example
        -------
        """
        visited = set()
        node2ancestors = defaultdict(set)
        node_queue = [node for node in self._nodes if not self._parents[node]]

        while node_queue:
            node = node_queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for child in self._children[node]:
                node2ancestors[child].add(node)
                node2ancestors[child].update(node2ancestors[node])
                node_queue.append(child)

        return core_utils.defdict2dict(node2ancestors, self._nodes)

    def descendants_of(self, node, exclude_arcs=set()):
        """Return the nodes downstream of node

        Parameters
        ----------
        node:
            The node.

        See Also
        --------
        ancestors_of

        Return
        ------
        Set[node]
            Return all nodes j such that there is a directed path from node j.

        Example
        -------
        """
        descendants = set()
        self._add_descendants(descendants, node, exclude_arcs=exclude_arcs)
        return descendants

    def has_directed(self, i, j):
        return (i, j) in self._directed

    def has_bidirected(self, i, j):
        return tuple(sorted((i, j))) in self._bidirected

    def has_undirected(self, i, j):
        return tuple(sorted((i, j))) in self._undirected

    def has_any_edge(self, i, j):
        return self.has_directed(i, j) or self.has_directed(j, i) or self.has_bidirected(i, j) or self.has_undirected(i, j)

    def vstructures(self):
        vstructs = set()
        for node in self._nodes:
            for p1, p2 in itr.combinations(self._parents[node] | self._spouses[node], 2):
                if not self.has_any_edge(p1, p2):
                    p1_, p2_ = sorted((p1, p2))
                    vstructs.add((p1_, node, p2))
        return vstructs

    def colliders(self):
        return {node for node in self._nodes if len(self._parents[node] | self._spouses[node]) >= 2}

    def _bidirected_reachable(self, node, tmp, visited):
        visited.add(node)
        tmp.add(node)
        for spouse in filter(lambda spouse: spouse not in visited, self._spouses[node]):
            tmp = self._bidirected_reachable(spouse, tmp, visited)
        return tmp

    def c_components(self):
        """Return the c-components of this graph.

        Return
        ------
        List[Set[node]]
            Return the partition of nodes coming from the relation of reachability by bidirected edges.
        """
        node_queue = self._nodes.copy()
        components = []
        visited_nodes = set()

        while node_queue:
            node = node_queue.pop()
            if node not in visited_nodes:
                components.append(self._bidirected_reachable(node, set(), visited_nodes))

        return components

    def district_of(self, node):
        """Return the district of a node, i.e., the set of nodes reachable by bidirected edges.

        Return
        ------
        Set[node]
            The district of node.
        """
        return self._bidirected_reachable(node, set(), set())

    def discriminating_paths(self):
        colliders = self.colliders()
        discriminating_paths = {}
        for j, parents in self._parents.items():  # potential endpoints of discriminating paths
            if not parents:
                break
            nonadjacent = self._nodes - parents - self._children[j] - self._spouses[j] - {j}
            for i in nonadjacent:  # potential start points of discriminating paths
                # search all paths that satisfy discriminating path criteria
                path_queue = [
                    [i, k]
                    for k in self._spouses[i] | self._children[i]
                    if k in colliders and j in self._children[k]
                ]
                while path_queue:
                    path = path_queue.pop(0)
                    final_node = path[-1]
                    # check if path is discriminating for the next node
                    for k in filter(lambda k: k not in path, self._spouses[final_node]):
                        if j in self._spouses[k]:
                            full_path = path.copy()
                            full_path.extend([k, j])
                            discriminating_paths[tuple(full_path)] = 'c'
                        elif j in self._children[k]:
                            full_path = path.copy()
                            full_path.extend([k, j])
                            discriminating_paths[tuple(full_path)] = 'n'
                    for k in filter(lambda k: k not in path, self._parents[final_node]):
                        if j in self._children[k]:
                            full_path = path.copy()
                            full_path.extend([k, j])
                            discriminating_paths[tuple(full_path)] = 'n'

                    # extend path
                    for k in self._spouses[final_node]:
                        if k not in path and k in colliders and j in self._children[k]:
                            new_path = path.copy()
                            new_path.append(k)
                            path_queue.append(new_path)
        return discriminating_paths

    def _reachable(self, start_node, end_node, visited=set(), allowed_edges={'b', 'u', 'c', 'p'}, predicate=lambda node: True):
        allowed_nbrs = set()
        if 'b' in allowed_edges:
            allowed_nbrs.update(self._spouses[start_node])
        if 'u' in allowed_edges:
            allowed_nbrs.update(self._neighbors[start_node])
        if 'c' in allowed_edges:
            allowed_nbrs.update(self._children[start_node])
        if 'p' in allowed_edges:
            allowed_nbrs.update(self._parents[start_node])

        allowed_nbrs = {nbr for nbr in allowed_nbrs if predicate(nbr)}

        results = []
        for nbr in allowed_nbrs:
            if nbr in visited:
                continue
            visited.add(nbr)
            if nbr == end_node:
                return True
            results.append(self._reachable(nbr, end_node, visited, allowed_edges, predicate))

        return any(results)

    # === ???
    def pairwise_markov_statements(self):
        statements = set()
        for i, j in itr.combinations(self._nodes, 2):
            if not self.has_any_edge(i, j):
                statements.add((i, j, frozenset(self.ancestors_of(i) | self.ancestors_of(j) - {i, j})))
        return statements

    def is_imap(self, other, certify=False):
        if not self.is_maximal():
            raise Exception("Your graph is not maximal")

        certificate = next(
            ((i, j, S) for i, j, S in self.pairwise_markov_statements() if not other.msep(i, j, S)),
            None)
        is_imap_ = certificate is None
        if certify: return is_imap_, certificate
        else: return is_imap_

    # def is_minimal_imap(self, other, certify=False):
    #     print("THIS HAS NOT BEEN TESTED")
    #     certificate = next((
    #         i, j for i, j in self._directed | self._bidirected
    #         if other.msep(i, j, self.ancestors_of(i) | self.ancestors_of(j) - {i, j})
    #     ), False)
    #     res = not certificate and self.is_imap(other)
    #     if not certify:
    #         return res
    #     else:
    #         return res, certificate

    def is_minimal_imap(self, other, certify=False, check_imap=True):
        if check_imap and not self.is_imap(other):
            return False, None

        for i, j in random.sample(list(self._directed)+list(self._bidirected), self.num_bidirected+self.num_directed):
            new_mag = self.copy()
            if self.has_bidirected(i, j):
                new_mag.remove_bidirected(i, j)
            if self.has_directed(i, j):
                new_mag.remove_directed(i, j)
            if new_mag.is_maximal() and new_mag.is_imap(other):
                if certify: return False, (i, j)
                else: return False
        if certify: return True, None
        else: return True

    def is_minimal_imap2(self, other, certify=False, check_imap=True, validate=False):
        if check_imap and not self.is_imap(other):
            return False, None

        for i, j in random.sample(list(self._directed)+list(self._bidirected), self.num_directed+self.num_bidirected):
            if other.msep(i, j, self.ancestors_of(i) | self.ancestors_of(j) - {i, j}):
                new_mag = self.copy()
                if self.has_bidirected(i, j):
                    new_mag.remove_bidirected(i, j)
                else:
                    new_mag.remove_directed(i, j)
                if new_mag.is_maximal():
                    if validate:
                        if not new_mag.is_imap(other):
                            raise Exception
                    if certify: return False, (i, j)
                    else: return False
        if certify: return True, None
        else: return True

    def is_minimal_imap3(self, other, certify=False, check_imap=True, validate=False, verbose=False):
        if check_imap and not self.is_imap(other):
            return False, None

        for i, j in random.sample(list(self._directed)+list(self._bidirected), self.num_directed+self.num_bidirected):
            new_mag = self.copy()
            if self.has_bidirected(i, j):
                new_mag.remove_bidirected(i, j)
            else:
                new_mag.remove_directed(i, j)
            current_markov_blanket = set.union(*(set(v) for v in self.markov_blanket(j).values())) | self.district_of(j)
            new_markov_blanket = set.union(*(set(v) for v in new_mag.markov_blanket(j).values())) | new_mag.district_of(j)
            mb_difference = (current_markov_blanket - new_markov_blanket - {j}) | {i}
            rest = new_markov_blanket - {i, j}
            if verbose: print(f'i={i}, j={j}, mb_diff={mb_difference}, rest={rest}')
            if verbose: print("H", self)
            if verbose: print("G", other)

            if other.msep(i, mb_difference, rest) and new_mag.is_maximal():
                print('here')
                if validate:
                    if not new_mag.is_imap(other):
                        raise Exception
                if certify: return False, (i, j)
                else: return False
        if certify: return True, None
        else: return True

    def is_minimal_imap4(self, other, certify=False, check_imap=True, validate=False, extra_validate=False, verbose=False):
        if check_imap and not self.is_imap(other):
            raise Exception("Not an IMAP")
            print("isn't imap")
            return False, None

        if extra_validate:
            for i, j in self._directed | self._bidirected:
                new_mag = self.copy()
                new_mag.remove_edge(i, j)
                s = new_mag.induced_subgraph(new_mag.ancestors_of({i, j}) | {i, j}).markov_blanket(j, flat=True) - {j}
                if other.msep(i, j, s) and new_mag.is_maximal():
                    if not new_mag.is_imap(other):
                        raise Exception("CI test not sufficient", new_mag, other, i, j, s)
            print('extra validated')

        for i, j in random.sample(list(self._directed)+list(self._bidirected), self.num_directed+self.num_bidirected):
            change = False
            new_mag = self.copy()
            new_mag.remove_edge(i, j)

            # works:
            set1 = (new_mag.markov_blanket(j, flat=True) & new_mag.ancestors_of(i)) | new_mag.parents_of(j)
            remove_edge = other.msep(i, j, set1)

            # new:
            set2 = new_mag.induced_subgraph(new_mag.ancestors_of({i, j}) | {i, j}).markov_blanket(j, flat=True) - {i, j}
            remove_edge2 = other.msep(i, j, set2)
            # print(i, j, set2)
            set3 = new_mag.markov_blanket(j, flat=True) & new_mag.ancestors_of({j}) - {j}

            # if set2 != set3:
            #     print(new_mag, j, set2, set3)

            if remove_edge2:
                change = True
            # if self.has_bidirected(i, j) and other.msep(i, j, self.parents_of(j)) and other.msep(i, j, self.parents_of(i)):
            #     new_mag = self.copy()
            #     new_mag.remove_bidirected(i, j)
            #     change = True
            # elif self.has_directed(i, j) and other.msep(i, j, self.parents_of(j) - {i}):
            #     new_mag = self.copy()
            #     new_mag.remove_directed(i, j)
            #     change = True
            if change and new_mag.is_maximal():
                if validate:
                    if not new_mag.is_imap(other):
                        raise Exception("CI test isn't sufficient: new MAG is not an IMAP")
                if certify: return False, (i, j)
                else: return False

        if certify: return True, None
        else: return True

    def markov_blanket(self, node, flat=False):
        """
        Return the Markov blanket of a node with respect to the whole graph.

        Parameters
        ----------
        node: The node whose Markov blanket to find.
        flat: if True, return the Markov blanket as a set, otherwise return a dictionary mapping nodes in the district
            of node to their parents.

        Returns
        -------
        The Markov blanket of node, including the node itself.
        """
        if not flat:
            return {d: self._parents[d] for d in self.district_of(node)}
        else:
            district = self.district_of(node)
            return district | set.union(*(self._parents[d] for d in district)) | {node}

    def resolved_quasisinks(self, other):
        res_qsinks = set()
        while True:
            new_resolved = {
                node for node in self._nodes - res_qsinks
                if not (self._children[node] - res_qsinks) and
                   not (other._children[node] - res_qsinks) and
                   self.markov_blanket(node) == other.markov_blanket(node)
            }
            res_qsinks.update(new_resolved)
            if not new_resolved:
                break

        return res_qsinks

    def is_maximal(self):
        new_mag = self.copy()
        new_mag.to_maximal()
        return new_mag == self

    def to_maximal(self, new=False):
        if new:
            # === NEED DICTIONARY OF ANCESTORS AND C-COMPONENTS TO CHECK INDUCING PATHS
            ancestor_dict = self.ancestor_dict()
            c_components = self.c_components()
            node2component = dict()
            for ix, component in enumerate(c_components):
                for node in component:
                    node2component[node] = ix

            # === FIND INDUCING PATHS BETWEEN PAIRS OF NODE
            induced_pairs = []

            non_adjacent_pairs = ((i, j) for i, j in itr.combinations(self._nodes, 2) if not self.has_any_edge(i, j))
            for node1, node2 in non_adjacent_pairs:
                check_ancestry = lambda node: node in ancestor_dict[node1] or node in ancestor_dict[node2]
                nbrs1 = self._children[node1] | self._spouses[node1]
                nbrs2 = self._children[node2] | self._spouses[node2]

                # ONLY CHECK PATHS BETWEEN SPOUSES/CHILDREN THAT ARE IN THE SAME C-COMPONENT
                for nbr1, nbr2 in itr.product(nbrs1, nbrs2):
                    same_component = node2component[nbr1] == node2component[nbr2]
                    if same_component and nbr1 in ancestor_dict[node2] and nbr2 in ancestor_dict[node1]:
                        if self._reachable(nbr1, nbr2, allowed_edges={'b'}, predicate=check_ancestry):
                            induced_pairs.append((node1, node2))
                            continue

            for node1, node2 in induced_pairs:
                self._add_bidirected(node1, node2)
        else:
            for i, j in itr.combinations(self._nodes, r=2):
                if not self.has_any_edge(i, j):
                    never_msep = not any(self.msep(i, j, S) for S in core_utils.powerset(self._nodes - {i, j}))
                    if never_msep: self.add_bidirected(i, j)

    def to_pag(self):
        raise NotImplementedError

    # === CONVERTERS
    def to_amat(self):
        amat = np.zeros([self.nnodes, self.nnodes])
        for i, j in self.directed:
            amat[i, j] = 2
            amat[j, i] = 3
        for i, j in self.bidirected:
            amat[i, j] = 2
            amat[j, i] = 2
        for i, j in self.undirected:
            amat[i, j] = 3
            amat[j, i] = 3
        return amat

    @staticmethod
    def from_amat(amat):
        p = amat.shape[0]
        directed = set()
        bidirected = set()
        undirected = set()
        for i, j in itr.combinations(set(range(p)), 2):
            vij = amat[i, j]
            vji = amat[j, i]
            if vij == 2 and vji == 3:
                directed.add((i, j))
            if vij == 3 and vji == 2:
                directed.add((j, i))
            if vij == 2 and vji == 2:
                bidirected.add((i, j))
            if vij == 3 and vji == 3:
                undirected.add((i, j))
        return AncestralGraph(set(range(p)), directed, bidirected, undirected)

    # === COMPARISON
    def markov_equivalent(self, other):
        same_skeleton = self.skeleton == other.skeleton
        same_vstructures = self.vstructures() == other.vstructures()

        self_discriminating_paths = self.discriminating_paths()
        other_discriminating_paths = other.discriminating_paths()
        shared_disc_paths = set(self_discriminating_paths.keys()) & set(other_discriminating_paths)
        same_discriminating = all(
            self_discriminating_paths[path] == other_discriminating_paths[path]
            for path in shared_disc_paths
        )

        return same_skeleton and same_vstructures and same_discriminating

    def shd_skeleton(self, other):
        return len(self.skeleton.symmetric_difference(other.skeleton))

    def as_hashed(self):
        return frozenset(self._directed), frozenset(self._bidirected), frozenset(self._undirected)

    # === Algorithms
    def _add_upstream(self, upstream, node):
        for parent in self._parents[node]:
            if parent not in upstream:
                upstream.add(parent)
                self._add_upstream(upstream, parent)

    def _is_collider(self, u, v, w):
        """return True if u-v-w is a collider"""
        if v in self._children[u] and v in self._children[w]:
            return True
        elif v in self._children[u] and v in self._spouses[w]:
            return True
        elif v in self._spouses[u] and v in self._children[w]:
            return True
        elif v in self._spouses[u] and v in self._spouses[w]:
            return True
        else:
            return False

    def _no_other_path(self, i, j, ancestor_dict):
        """
        Check if there is any path from i to j other than possibly the direct edge i->j.
        """
        other_ancestors_j = ancestor_dict[j] - {i}
        return other_ancestors_j & self._children[i] == set()

    def legitimate_mark_changes(self, verbose=False, strict=True):
        """
        Return directed edges that can be changed to bidirected edges, and bidirected edges that can be changed to
        directed edges.

        Parameters
        ----------
        strict: If True, check discriminating path condition. Otherwise, check only equality of parents and spouses.

        Return
        ------
        (mark_changes_dir, mark_changes_bidir)
            Directed edges that can be changed to bidirected edges, and bidirected edges that can be changed to directed
            edges (which will be the new directed edge).

        Example
        -------
        >>> g = cd.AncestralGraph(directed={(0, 1)}, bidirected={(1, 2)})
        >>> g.legitimate_mark_changes()
        ({(0, 1)}, {(2, 1)})
        """
        if self._undirected:
            raise ValueError('Only defined for DMAGs')

        if not strict:
            # print("TODO: CHECK")
            ancestor_dict = self.ancestor_dict()
            mark_changes_dir = {
                (i, j) for i, j in self._directed
                if self._parents[i] - self._parents[j] == set() and
                   self._spouses[i] - self._parents[j] - self._spouses[j] == set()
                   and self._no_other_path(i, j, ancestor_dict)
            }
            mark_changes_bidir = {
                (i, j) for i, j in self._bidirected
                if self._parents[i] - self._parents[j] == set() and
                   self._spouses[i] - self._parents[j] - self._spouses[j] == set()
            }
            return mark_changes_dir, mark_changes_bidir

        if strict:
            disc_paths = self.discriminating_paths()
            ancestor_dict = self.ancestor_dict()

            mark_changes_dir = set()
            for i, j in self._directed:
                if verbose: print(f'{i}->{j} => {i}<->{j} ?')
                parents_condition = self._parents[i] - self._parents[j] == set()
                if not parents_condition: continue
                spouses_condition = self._spouses[i] - self._spouses[j] - self._parents[j] == set()
                if not spouses_condition: continue
                ancestral_condition = self._no_other_path(i, j, ancestor_dict)
                if not ancestral_condition: continue

                # SECOND CONDITION
                disc_paths_for_i = [path for path in disc_paths.keys() if path[-2] == i]
                disc_paths_condition = all(path[-1] != j for path in disc_paths_for_i)
                if not disc_paths_condition:
                    if verbose: print('Failed discriminating path condition')
                    continue

                if verbose: print('Passed')
                mark_changes_dir.add((i, j))

            mark_changes_bidir = set()
            for i, j in self._bidirected | set(map(reversed, self._bidirected)):
                if verbose: print(f'{i}<->{j} => {i}->{j} ?')
                parents_condition = self._parents[i] - self._parents[j] == set()
                if not parents_condition: continue
                spouses_condition = self._spouses[i] - {i} - self._spouses[j] - self._parents[j] == set()
                if not spouses_condition: continue
                ancestral_condition = self._no_other_path(i, j, ancestor_dict)
                if not ancestral_condition: continue

                # SECOND CONDITION
                disc_paths_for_i = [path for path in disc_paths.keys() if path[-2] == i]
                disc_paths_condition = all(path[-1] != j for path in disc_paths_for_i)
                if not disc_paths_condition:
                    if verbose: print('Failed discriminating path condition')
                    continue

                if verbose: print('Passed')
                mark_changes_bidir.add((i, j))

            return mark_changes_dir, mark_changes_bidir

    def msep(self, A, B, C=set()):
        """
        Check whether A and B are m-separated given C, using the Bayes ball algorithm.


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
        # marks whether the node has been encountered along a path where it has a tail or an arrowhead
        _t = 'tail'  # tail
        _a = 'arrowhead'  # arrowhead

        schedule = {(node, _t) for node in A}
        while schedule:
            node, _dir = schedule.pop()
            if node in B: return False
            if (node, _dir) in visited: continue
            visited.add((node, _dir))
            # print(node, _dir)

            # if coming through a tail, won't encounter v-structure
            if _dir == _t and node not in C:
                schedule.update({(parent, _t) for parent in self._parents[node]})
                schedule.update({(child, _a) for child in self._children[node]})
                schedule.update({(spouse, _a) for spouse in self._spouses[node]})
                schedule.update({(nbr, _t) for nbr in self._neighbors[node]})

            if _dir == _a:
                # if coming through an arrowhead and see shaded node, can go through v-structure
                if node in shaded_nodes:
                    schedule.update({(parent, _t) for parent in self._parents[node]})
                    schedule.update({(spouse, _a) for spouse in self._spouses[node]})

                # if coming through an arrowhead and see unconditioned node, can go through children and neighbors
                if node not in C:
                    schedule.update({(child, _a) for child in self._children[node]})
                    schedule.update({(nbr, _a) for nbr in self._neighbors[node]})

        return True

    def msep_from_given(self, A, C=set()):
        """Find all nodes m-seperated from A given C using algorithm similar to that in Geiger, D., Verma, T., & Pearl, J. (1990). Identifying independence in Bayesian networks. Networks, 20(5), 507-534."""

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
            # Find all unlabled links v->w adjacent to at least one link u->v labeled i, such that (u->v,v->w) is a legal pair.
            for link in i_links:
                u, v = link
                for w in self._adjacent[v]:
                    if not u == w and (v, w) not in labeled_links:
                        if self._is_collider(u, v, w):  # Is collider?
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


if __name__ == '__main__':
    g = AncestralGraph(nodes=set(range(1, 5)), directed={(1, 2), (2, 4), (3, 2), (3, 4)})
    disc_paths = g.discriminating_paths()



