from collections import defaultdict


class CycleError(Exception):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        message = '%s -> %s will cause a cycle' % (source, target)
        super().__init__(message)


class SpouseError(Exception):
    def __init__(self, ancestor, desc):
        self.ancestor = ancestor
        self.desc = desc
        message = '%s <-> %s cannot be added since %s is an ancestor of %s' % (ancestor, desc, ancestor, desc)
        super().__init__(message)


class AdjacentError(Exception):
    def __init__(self, node1, node2, arrow_type):
        self.node1 = node1
        self.node2 = node2
        self.arrow_type = arrow_type
        message = '%s %s %s cannot be added since %s and %s are already adjacent' % (node1, arrow_type, node2, node1, node2)
        super().__init__(message)


class NeighborError(Exception):
    def __init__(self, node, neighbors=None, parents=None, spouses=None):
        self.node = node
        self.neighbors = neighbors
        self.parents = parents
        self.spouses = spouses
        if self.neighbors:
            message = 'The node %s has neighbors %s. Nodes cannot have neighbors and parents/spouses.' % (node, ','.join(map(str, neighbors)))
        elif self.parents:
            message = 'The node %s has parents %s. Nodes cannot have neighbors and parents/spouses.' % (node, ','.join(map(str, parents)))
        elif self.spouses:
            message = 'The node %s has spouses %s. Nodes cannot have neighbors and parents/spouses.' % (node, ','.join(map(str, spouses)))
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

    def copy(self):
        """
        Return a copy of the current ancestral graph.
        """
        return AncestralGraph(self.nodes, self.directed, self.bidirected, self.undirected)

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

    def add_directed(self, i, j):
        self._add_directed(i, j)

    def add_bidirected(self, i, j):
        self._add_bidirected(i, j)

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

    # === PROPERTIES
    @property
    def nodes(self):
        return self._nodes.copy()

    @property
    def directed(self):
        return self._directed.copy()

    @property
    def bidirected(self):
        return self._bidirected.copy()

    @property
    def undirected(self):
        return self._undirected.copy()

    def children_of(self, i):
        return self._children[i].copy()

    def parents_of(self, i):
        return self._parents[i].copy()

    def spouses_of(self, i):
        return self._spouses[i].copy()

    def neighbors_of(self, i):
        return self._neighbors[i].copy()

    def ancestors_of(self, i):
        raise NotImplementedError

    def descendants_of(self, i):
        raise NotImplementedError

    def has_directed(self, i, j):
        return (i, j) in self._directed

    def has_bidirected(self, i, j):
        return tuple(sorted((i, j))) in self._bidirected

    def has_undirected(self, i, j):
        return tuple(sorted((i, j))) in self._undirected

    # === ???
    def to_maximal(self):
        raise NotImplementedError

    def to_pag(self):
        raise NotImplementedError

    # === CONVERTERS
    def to_amat(self):
        raise NotImplementedError

    def from_amat(self):
        raise NotImplementedError


