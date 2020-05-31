from causaldag.utils.core_utils import to_set
from causaldag.classes.custom_types import Node
from typing import Set, Union, List
from collections import defaultdict


class Distribution:
    def __init__(self, marginal, cond_set=set(), intervened=set()):
        self.marginal = marginal
        self.cond_set = to_set(cond_set)
        self.intervened = to_set(intervened)

    def __str__(self):
        return f'P({self.marginal} | do({self.intervened}), {self.cond_set})'


class MarginalDistribution:
    def __init__(self, distribution, marginalized):
        self.distribution = distribution
        self.marginalized = marginalized

    def __str__(self):
        return f'\sum_{self.marginalized} {self.distribution}'


class ConditionalDistribution:
    def __init__(self, distribution, cond_set):
        self.distribution = distribution
        self.cond_set = to_set(cond_set)

    def __str__(self):
        return f'P()'


class ProductDistribution:
    def __init__(self, terms):
        self.terms = terms

    def __str__(self):
        return '*'.join((str(term) for term in self.terms))


class NonIdentifiabilityError(RuntimeError):
    def __init__(self):
        pass


class SemiMarkovian:
    def __init__(self, nodes, directed: set(), bidirected: set()):
        self._nodes = nodes.copy()
        self._directed = set()
        self._bidirected = set()
        self._undirected = set()

        self._spouses = defaultdict(set)
        self._parents = defaultdict(set)
        self._children = defaultdict(set)
        self._adjacent = defaultdict(set)

        for i, j in directed:
            self._add_directed(i, j)
        for i, j in bidirected:
            self._add_bidirected(i, j)

    def _add_directed(self, i: Node, j: Node):
        self._nodes.add(i)
        self._nodes.add(j)
        self._directed.add((i, j))
        self._parents[j].add(i)
        self._children[i].add(j)

        self._adjacent[i].add(j)
        self._adjacent[j].add(i)

    def _add_bidirected(self, i: Node, j: Node):
        self._nodes.add(i)
        self._nodes.add(j)
        self._bidirected.add(frozenset({i, j}))
        self._spouses[j].add(i)
        self._spouses[i].add(j)

        self._adjacent[i].add(j)
        self._adjacent[j].add(i)

    def parents_of(self, nodes):
        return set.union(*(self._parents[node] for node in nodes))

    def _add_ancestors(self, ancestors: Set, nodes: Set[Node]):
        parents = self.parents_of(nodes)
        new_ancestors = parents - ancestors
        ancestors.update(new_ancestors)
        self._add_ancestors(ancestors, new_ancestors)

    def ancestors_of(self, node: Union[Node, Set[Node]]) -> Set[Node]:
        ancestors = set()
        self._add_ancestors(ancestors, {node})
        return ancestors

    def mutilated_graph(self, outgoing_mutilated=None, incoming_mutilated=None):
        remaining_directed = {
            (i, j) for i, j in self._directed
            if i not in outgoing_mutilated and j not in incoming_mutilated
        }
        return SemiMarkovian(self._nodes, remaining_directed, self._bidirected)

    def induced_subgraph(self, nodes=set(), removed_nodes=None):
        nodes = self._nodes - removed_nodes if removed_nodes is not None else nodes

        new_directed = {(i, j) for i, j in self._directed if {i, j} <= nodes}
        new_bidirected = {(i, j) for i, j in self._bidirected if {i, j} <= nodes}
        return SemiMarkovian(nodes, new_directed, new_bidirected)

    def _bidirected_reachable(self, node, tmp: Set[Node], visited: Set[Node]) -> Set[Node]:
        visited.add(node)
        tmp.add(node)
        for spouse in filter(lambda spouse: spouse not in visited, self._spouses[node]):
            tmp = self._bidirected_reachable(spouse, tmp, visited)
        return tmp

    def c_components(self) -> List[set]:
        node_queue = self._nodes.copy()
        components = []
        visited_nodes = set()

        while node_queue:
            node = node_queue.pop()
            if node not in visited_nodes:
                components.append(self._bidirected_reachable(node, set(), visited_nodes))

        return components

    def general_identification(self, y, x, available_experiments):
        x, y = to_set(x), to_set(y)

        # LINE 2
        matching_experiment = next((e for e in available_experiments if x == e), None)
        if matching_experiment:
            return Distribution(y, intervened=matching_experiment | x)

        # LINE 3
        ancestors_y = self.ancestors_of(y)
        if ancestors_y != self._nodes:
            new_graph = self.induced_subgraph(ancestors_y)
            return new_graph.general_identification(y, x & ancestors_y, available_experiments)

        # LINE 4
        x_mutilated_graph = self.mutilated_graph(incoming_mutilated=x)
        ancestors_y_no_x = x_mutilated_graph.ancestors_of(y)
        w = self._nodes - x - ancestors_y_no_x
        if len(w) != 0:
            return self.general_identification(y, x | w, available_experiments)

        # LINE 5
        g_minus_x = self.induced_subgraph(removed_nodes=x)
        s_components = g_minus_x.c_components()

        # LINE 6
        if len(s_components) > 1:
            prod = ProductDistribution([
                self.general_identification(s, self._nodes - s, available_experiments)
                for s in s_components
            ])
            return MarginalDistribution(prod, self._nodes - x - y)

        # LINE 7
        for z in available_experiments:
            if z <= x:
                g_no_zx = self.induced_subgraph(removed_nodes=z|x)
                distribution = Distribution(self._nodes - z - x, intervened=z)  # TODO right distribution?
                sub_id = g_no_zx.sub_identification(y, x - z, distribution)
                if sub_id is not None:
                    return sub_id

        # LINE 8
        raise NonIdentifiabilityError

    def sub_identification(self, y, x, distribution):
        # LINE 11
        if len(x) == 0:
            return MarginalDistribution(distribution, self._nodes - y)

        # LINE 12
        ancestors_y = self.ancestors_of(y)
        if ancestors_y != self._nodes:
            new_graph = self.induced_subgraph(ancestors_y)
            distribution = MarginalDistribution(distribution, self._nodes - ancestors_y)
            return new_graph.sub_identification(y, x & ancestors_y, distribution)

        # LINE 13
        c_components = self.c_components()
        if len(c_components) == 1:
            return None

        # LINE 10
        g_no_x = self.induced_subgraph(removed_nodes=x)
        s_components = g_no_x.c_components()
        assert len(s_components) == 1
        s = s_components[0]

        # LINE 14
        if any(c == s for c in c_components):
            prod = ProductDistribution()
            return MarginalDistribution(prod, s - y)

        # LINE 15
        for c in c_components:
            if s < c:
                new_graph = self.induced_subgraph(c)
                prod = ProductDistribution([])
                return new_graph.sub_identification(y, x & c, prod)

