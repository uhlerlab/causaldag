from collections import defaultdict
from causaldag.utils import core_utils
import itertools as itr


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

    def copy(self):
        """
        Return a copy of the current ancestral graph.
        """
        return AncestralGraph(self.nodes, self.directed, self.bidirected, self.undirected)

    def __str__(self):
        return 'Directed edges: %s, Bidirected edges: %s, Undirected edges: %s' % (self._directed, self._bidirected, self._undirected)

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
        return {tuple(sorted((i, j))) for i, j in self._bidirected | self._undirected | self._directed}

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

    def discriminating_paths(self):
        colliders = self.colliders()
        discriminating_paths = []
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
                            discriminating_paths.append((full_path, 'c'))
                        elif j in self._children[k]:
                            full_path = path.copy()
                            full_path.extend([k, j])
                            discriminating_paths.append((full_path, 'n'))
                    for k in filter(lambda k: k not in path, self._parents[final_node]):
                        if j in self._children[k]:
                            full_path = path.copy()
                            full_path.extend([k, j])
                            discriminating_paths.append((full_path, 'n'))

                    # extend path
                    for k in self._spouses[final_node]:
                        if k not in path and k in colliders and j in self._children[k]:
                            new_path = path.copy()
                            new_path.append(k)
                            path_queue.append(new_path)
        return discriminating_paths

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

    # === COMPARISON
    def markov_equivalent(self, other):
        same_skeleton = self.skeleton == other.skeleton
        same_vstructures = self.vstructures() == other.vstructures()
        same_discriminating = self.discriminating_paths() == other.discriminating_paths()
        return same_skeleton and same_vstructures and same_discriminating

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



