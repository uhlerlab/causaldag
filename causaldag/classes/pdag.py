# Author: Chandler Squires
"""
Base class for partially directed acyclic graphs
"""

from collections import defaultdict
from causaldag.utils import core_utils
import itertools as itr
import numpy as np
from typing import Set
from collections import namedtuple
from scipy.special import factorial
import networkx as nx
from typing import Set, FrozenSet

SmallDag = namedtuple('SmallDag', ['arcs', 'reversible_arcs', 'parents_dict', 'children_dict', 'level'])


class PDAG:
    def __init__(
            self,
            nodes: Set=set(),
            arcs: Set=set(),
            edges: Set=set(),
            known_arcs=set()
    ):
        self._nodes = set(nodes)
        self._arcs = set()
        self._edges = set()
        self._parents = defaultdict(set)
        self._children = defaultdict(set)
        self._neighbors = defaultdict(set)
        self._undirected_neighbors = defaultdict(set)
        for i, j in arcs:
            self._add_arc(i, j)
        for i, j in edges:
            self._add_edge(i, j)

        self._known_arcs = known_arcs.copy()

    @classmethod
    def from_amat(cls, amat):
        """Return a PDAG with arcs/edges given by amat
        """
        nrows, ncols = amat.shape
        arcs = set()
        edges = set()
        for (i, j), val in np.ndenumerate(amat):
            if val != 0:
                if (j, i) in arcs:
                    arcs.remove((j, i))
                    edges.add((i, j))
                else:
                    arcs.add((i, j))
        return PDAG(set(range(nrows)), arcs, edges)

    def to_nx(self):
        if self._arcs:
            raise NotImplementedError
        g = nx.Graph()
        g.add_edges_from(self._edges)
        return g

    @classmethod
    def from_nx(cls, nx_graph):
        return PDAG(nodes=nx_graph.nodes, edges=nx_graph.edges)

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

    def copy(self):
        """Return a copy of the graph
        """
        return PDAG(nodes=self._nodes, arcs=self._arcs, edges=self._edges, known_arcs=self._known_arcs)

    # === PROPERTIES
    @property
    def nodes(self):
        return set(self._nodes)

    @property
    def nnodes(self):
        return len(self._nodes)

    @property
    def num_arcs(self):
        return len(self._arcs)

    @property
    def num_edges(self):
        return len(self._edges)

    @property
    def num_adjacencies(self):
        return self.num_arcs + self.num_edges

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

    @property
    def skeleton(self):
        return {frozenset({i, j}) for i, j in self._arcs | self._edges}

    @property
    def dominated_nodes(self):
        dominated_nodes = set()
        for node in self._nodes:
            num_nbrs = self.undirected_degree_of(node)
            if num_nbrs == 0:
                dominated_nodes.add(node)
            elif num_nbrs == 1:
                max_nbrs_of_nbrs = max(len(self._undirected_neighbors[nbr]) for nbr in self._undirected_neighbors[node])
                if max_nbrs_of_nbrs > 1:
                    dominated_nodes.add(node)
        return dominated_nodes

    def clique_size(self):
        if len(self._arcs) == 0:
            g = self.to_nx()
            return nx.chordal_graph_treewidth(g) + 1
        else:
            return max(cc.clique_size() for cc in self.chain_components())

    def max_cliques(self):
        if len(self._arcs) == 0:
            g = self.to_nx()
            m_cliques = nx.chordal_graph_cliques(g)
            return {frozenset(c) for c in m_cliques}
        else:
            return set.union(*(cc.max_cliques() for cc in self.chain_components()))

    # === PROPERTIES W/ ARGUMENTS
    def indegree_of(self, node):
        return len(self._parents[node])

    def outdegree_of(self, node):
        return len(self._children[node])

    def undirected_degree_of(self, node):
        return len(self._undirected_neighbors[node])

    def total_degree_of(self, node):
        return len(self._neighbors[node])

    def parents_of(self, node):
        return set(self._parents[node])

    def children_of(self, node):
        return set(self._children[node])

    def neighbors_of(self, node):
        return set(self._neighbors[node])

    def undirected_neighbors_of(self, node):
        return set(self._undirected_neighbors[node])

    def has_edge(self, i, j):
        """Return True if the graph contains the edge i--j
        """
        return frozenset({i, j}) in self._edges

    def has_arc(self, i, j):
        """Return True if the graph contains the arc i->j"""
        return (i, j) in self._arcs

    def has_edge_or_arc(self, i, j):
        """Return True if the graph contains the edge i--j or an arc i->j or i<-j
        """
        return (i, j) in self._arcs or (j, i) in self._arcs or self.has_edge(i, j)

    def vstructs(self):
        vstructs = set()
        for node in self._nodes:
            for p1, p2 in itr.combinations(self._parents[node], 2):
                if p1 not in self._parents[p2] and p2 not in self._parents[p1]:
                    vstructs.add((p1, node))
                    vstructs.add((p2, node))
        return vstructs

    def _undirected_reachable(self, node, tmp, visited):
        visited.add(node)
        tmp.add(node)
        for nbr in filter(lambda nbr: nbr not in visited, self._undirected_neighbors[node]):
            tmp = self._undirected_reachable(nbr, tmp, visited)
        return tmp

    def chain_components(self, rename=False):
        """Return the chain components of this graph.

        Return
        ------
        List[Set[node]]
            Return the partition of nodes coming from the relation of reachability by undirected edges.
        """
        node_queue = self._nodes.copy()
        components = []
        visited_nodes = set()

        while node_queue:
            node = node_queue.pop()
            if node not in visited_nodes:
                reachable = self._undirected_reachable(node, set(), visited_nodes)
                if len(reachable) > 1: components.append(reachable)

        return [self.induced_subgraph(c, rename=rename) for c in components]

    def induced_subgraph(self, nodes, rename=False):
        if rename:
            ixs = dict(map(reversed, enumerate(nodes)))
            new_nodes = set(range(len(nodes)))
            arcs = {(ixs[i], ixs[j]) for i, j in self._arcs if i in nodes and j in nodes}
            edges = {(ixs[i], ixs[j]) for i, j in self._edges if i in nodes and j in nodes}
        else:
            new_nodes = nodes
            arcs = {(i, j) for i, j in self._arcs if i in nodes and j in nodes}
            edges = {(i, j) for i, j in self._edges if i in nodes and j in nodes}
        return PDAG(nodes=new_nodes, arcs=arcs, edges=edges)

    def interventional_cpdag(self, dag, intervened_nodes):
        cut_edges = set()
        for node in intervened_nodes:
            cut_edges.update(dag.incident_arcs(node))
        p = PDAG(self._nodes, self._arcs | cut_edges, self._edges - {frozenset({i, j}) for i, j in cut_edges})
        p.to_complete_pdag()
        return p

    # === MUTATORS
    def _add_arc(self, i, j):
        self._nodes.add(i)
        self._nodes.add(j)
        self._arcs.add((i, j))

        self._neighbors[i].add(j)
        self._neighbors[j].add(i)

        self._children[i].add(j)
        self._parents[j].add(i)

    def _add_edge(self, i, j):
        self._nodes.add(i)
        self._nodes.add(j)
        self._edges.add(frozenset({i, j}))

        self._neighbors[i].add(j)
        self._neighbors[j].add(i)

        self._undirected_neighbors[i].add(j)
        self._undirected_neighbors[j].add(i)

    def remove_edge(self, i, j, ignore_error=False):
        try:
            self._edges.remove(frozenset({i, j}))
            self._neighbors[i].remove(j)
            self._neighbors[j].remove(i)
            self._undirected_neighbors[i].remove(j)
            self._undirected_neighbors[j].remove(i)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_node(self, node):
        """Remove a node from the graph
        """
        self._nodes.remove(node)
        self._arcs = {(i, j) for i, j in self._arcs if i != node and j != node}
        self._edges = {frozenset({i, j}) for i, j in self._edges if i != node and j != node}
        for child in self._children[node]:
            self._parents[child].remove(node)
            self._neighbors[child].remove(node)
        for parent in self._parents[node]:
            self._children[parent].remove(node)
            self._neighbors[parent].remove(node)
        for u_nbr in self._undirected_neighbors[node]:
            self._undirected_neighbors[u_nbr].remove(node)
            self._neighbors[u_nbr].remove(node)
        del self._parents[node]
        del self._children[node]
        del self._neighbors[node]
        del self._undirected_neighbors[node]

    def replace_edge_with_arc(self, arc, ignore_error=False):
        try:
            self._replace_edge_with_arc(arc)
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def _replace_arc_with_edge(self, arc):
        self._arcs.remove(arc)
        self._edges.add(frozenset({*arc}))
        i, j = arc
        self._parents[j].remove(i)
        self._children[i].remove(j)
        self._undirected_neighbors[i].add(j)
        self._undirected_neighbors[j].add(i)

    def _replace_edge_with_arc(self, arc):
        self._edges.remove(frozenset({*arc}))
        self._arcs.add(arc)
        i, j = arc
        self._parents[j].add(i)
        self._children[i].add(j)
        self._undirected_neighbors[i].remove(j)
        self._undirected_neighbors[j].remove(i)

    def assign_parents(self, node, parents, verbose=False):
        for p in parents:
            self._replace_edge_with_arc((p, node))
        for c in self._undirected_neighbors[node] - parents:
            self._replace_edge_with_arc((node, c))
        self.to_complete_pdag(verbose=verbose)

    def to_complete_pdag(self, verbose=False, solve_conflict=False):
        """
        Replace with arcs those edges whose orientations can be determined by Meek rules:
        =====
        See Koller & Friedman, Algorithm 3.5

        """
        if solve_conflict:
            raise NotImplementedError

        PROTECTED = 'P'  # indicates that some configuration definitely exists to protect the edge
        UNDECIDED = 'U'  # indicates that some configuration exists that could protect the edge
        NOT_PROTECTED = 'N'  # indicates no possible configuration that could protect the edge

        edges1 = {(i, j) for i, j in self._edges}
        undecided_arcs = edges1 | {(j, i) for i, j in edges1}
        arc_flags = {arc: PROTECTED for arc in self._arcs}
        arc_flags.update({arc: UNDECIDED for arc in undecided_arcs})

        while undecided_arcs:
            for arc in undecided_arcs:
                i, j = arc
                flag = NOT_PROTECTED

                # check configuration (a) -- causal chain
                s = ''
                for k in self._parents[i]:
                    if not self.has_edge_or_arc(k, j):
                        if arc_flags[(k, i)] == PROTECTED:
                            flag = PROTECTED
                            s = f': {k}->{i}-{j}'
                            break
                        else:
                            flag = UNDECIDED
                if verbose: print(f'{arc} marked {flag} by (a){s}')

                # check configuration (b) -- acyclicity
                s = ''
                if flag != PROTECTED:
                    for k in self._parents[j]:
                        if i in self._parents[k]:
                            if arc_flags[(i, k)] == PROTECTED and arc_flags[(k, j)] == PROTECTED:
                                flag = PROTECTED
                                s = f': {k}->{j}-{i}->{k}'
                                break
                            else:
                                flag = UNDECIDED
                    if verbose: print(f'{arc} marked {flag} by (b){s}')

                # check configuration (d)
                s = ''
                if flag != PROTECTED:
                    for k1, k2 in itr.combinations(self._parents[j], 2):
                        if self.has_edge(i, k1) and self.has_edge(i, k2) and not self.has_edge_or_arc(k1, k2):
                            if arc_flags[(k1, j)] == PROTECTED and arc_flags[(k2, j)] == PROTECTED:
                                flag = PROTECTED
                                s = f': {i}-{k1}->{j}<-{k2}-{i}'
                                break
                            else:
                                flag = UNDECIDED
                    if verbose: print(f'{arc} marked {flag} by (c){s}')

                arc_flags[arc] = flag

            if all(arc_flags[arc] == NOT_PROTECTED for arc in undecided_arcs): break

            for arc in undecided_arcs.copy():
                if arc_flags[arc] == PROTECTED:
                    if not solve_conflict:
                        if self.has_arc(arc[1], arc[0]):  # arc has already been oriented the opposite way
                            continue
                    undecided_arcs.remove(arc)
                    undecided_arcs.remove((arc[1], arc[0]))
                    self._replace_edge_with_arc(arc)

    def remove_unprotected_orientations(self, verbose=False):
        """
        Replace with edges those arcs whose orientations cannot be determined by either:
        - prior knowledge, or
        - Meek rules

        =====
        See Koller & Friedman, Algorithm 3.5

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

    def add_known_arc(self, i, j):
        if (i, j) in self._known_arcs:
            return
        self._known_arcs.add((i, j))
        self._edges.remove(frozenset({i, j}))
        self.remove_unprotected_orientations()

    def add_known_arcs(self, arcs):
        raise NotImplementedError

    # === MUTATORS
    def to_amat(self, node_list=None, mode='dataframe'):
        """Return an adjacency matrix for the graph
        """
        if node_list is None:
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
        for i, j in self._edges:
            amat[node2ix[i], node2ix[j]] = 1
            amat[node2ix[j], node2ix[i]] = 1

        if mode == 'dataframe':
            from pandas import DataFrame
            return DataFrame(amat, index=node_list, columns=node_list)
        else:
            return amat, node_list

    def _possible_sinks(self):
        return {node for node in self._nodes if len(self._children[node]) == 0}

    def _neighbors_covered(self, node):
        return {node2: self.neighbors[node2] - {node} == self.neighbors[node] for node2 in self._nodes}

    def to_dag(self):
        """
        Return a DAG that is consistent with this CPDAG.

        Returns
        -------
        d

        Examples
        --------
        TODO
        """
        from causaldag import DAG

        pdag2 = self.copy()
        arcs = set()
        while len(pdag2._edges) + len(pdag2._arcs) != 0:
            is_sink = lambda n: len(pdag2._children[n]) == 0
            no_vstructs = lambda n: all(
                (pdag2._neighbors[n] - {u_nbr}).issubset(pdag2._neighbors[u_nbr])
                for u_nbr in pdag2._undirected_neighbors[n]
            )
            sink = next((n for n in pdag2._nodes if is_sink(n) and no_vstructs(n)), None)
            if sink is None:
                break
            arcs.update((nbr, sink) for nbr in pdag2._neighbors[sink])
            pdag2.remove_node(sink)

        return DAG(arcs=arcs)

    # === MEC
    def mec_size(self):
        """Return the number of DAGs in the MEC represented by this PDAG
        """
        raise NotImplementedError

        if self.num_arcs != 0:
            pass
        elif self.num_edges == nnodes - 1:
            return nnodes
        elif self.num_edges == nnodes*(nnodes-1)/2:
            return factorial(nnodes)
        else:
            pass

    def exact_sample(self, save_sampler=True, nsamples=1):
        """Return a DAG sampled uniformly at random from the MEC represented by this PDAG
        """
        raise NotImplementedError

    def all_dags(self, verbose=False):
        """Return all DAGs consistent with this PDAG
        """
        dag = self.to_dag()
        arcs = dag._arcs
        all_arcs = set()

        orig_reversible_arcs = dag.reversible_arcs() - self._known_arcs
        orig_parents_dict = dag.parents
        orig_children_dict = dag.children

        level = 0
        q = [SmallDag(arcs, orig_reversible_arcs, orig_parents_dict, orig_children_dict, level)]
        while q:
            dag = q.pop()
            all_arcs.add(frozenset(dag.arcs))
            for i, j in dag.reversible_arcs:
                new_arcs = frozenset({arc for arc in dag.arcs if arc != (i, j)} | {(j, i)})
                if new_arcs not in all_arcs:
                    new_parents_dict = {}
                    new_children_dict = {}
                    for node in dag.parents_dict.keys():
                        parents = set(dag.parents_dict[node])
                        children = set(dag.children_dict[node])
                        if node == i:
                            new_parents_dict[node] = parents | {j}
                            new_children_dict[node] = children - {j}
                        elif node == j:
                            new_parents_dict[node] = parents - {i}
                            new_children_dict[node] = children | {i}
                        else:
                            new_parents_dict[node] = parents
                            new_children_dict[node] = children

                    new_reversible_arcs = dag.reversible_arcs.copy()
                    for k in dag.parents_dict[j]:
                        if (new_parents_dict[j] - {k}) == new_parents_dict[k] and (k, j) not in self._known_arcs:
                            new_reversible_arcs.add((k, j))
                        else:
                            new_reversible_arcs.discard((k, j))
                    for k in dag.children_dict[j]:
                        if new_parents_dict[j] == (new_parents_dict[k] - {j}) and (j, k) not in self._known_arcs:
                            new_reversible_arcs.add((j, k))
                        else:
                            new_reversible_arcs.discard((j, k))
                    for k in dag.parents_dict[i]:
                        if (new_parents_dict[i] - {k}) == new_parents_dict[k] and (k, i) not in self._known_arcs:
                            new_reversible_arcs.add((k, i))
                        else:
                            new_reversible_arcs.discard((k, i))
                    for k in dag.children_dict[i]:
                        if new_parents_dict[i] == (new_parents_dict[k] - {i}) and (i, k) not in self._known_arcs:
                            new_reversible_arcs.add((i, k))
                        else:
                            new_reversible_arcs.discard((i, k))

                    q.append(SmallDag(new_arcs, new_reversible_arcs, new_parents_dict, new_children_dict, dag.level+1))

        return all_arcs

    # === COMPARISON
    def shd(self, other):
        """Return the structural Hamming distance between this PDAG and another.

        For each pair of nodes, the SHD is incremented by 1 if the edge type/presence between the two nodes is different
        """
        self_undirected = {frozenset({*arc}) for arc in self._arcs} | self._edges
        other_undirected = {frozenset({*arc}) for arc in other._arcs} | other._edges
        num_additions = len(self_undirected - other_undirected)
        num_deletions = len(other_undirected - self_undirected)
        diff_type = {
            (i, j) for i, j in self_undirected & other_undirected
            if ((i, j) in self._arcs and (i, j) not in other._arcs) or
               ((j, i) in self._arcs and (j, i) not in other._arcs) or
               (frozenset({i, j}) in self._edges and frozenset({i, j}) not in other._edges)
        }
        return num_additions + num_deletions + len(diff_type)

    def shd_skeleton(self, other) -> int:
        return len(self.skeleton.symmetric_difference(other.skeleton))


if __name__ == '__main__':
    from causaldag.rand import directed_erdos

    g = directed_erdos(10, .5)
    c = g.cpdag()
    a1 = c.to_amat()
    a2, _ = c.to_amat(mode='numpy')
    a3, _ = c.to_amat(mode='sparse')
