from causaldag.utils.core_utils import to_set
from causaldag.classes.custom_types import Node
from typing import Set, Union, List, Optional
from collections import defaultdict
from operator import xor


class IDFormula:
    def __init__(self):
        self.active_variables = set()

    def get_conditional(self, marginal=None, cond_set=None):
        marginal = to_set(marginal)
        cond_set = to_set(cond_set)
        assert len(marginal & cond_set) == 0
        assert marginal | cond_set <= self.active_variables

    def marginalize_out(self, marginalized_variables: set):
        return self.get_conditional(self.active_variables-marginalized_variables)


class ProbabilityTerm(IDFormula):
    def __init__(self, active_variables: set, cond_set=None, intervened=None):
        super().__init__()

        self.active_variables = active_variables
        self.cond_set = to_set(cond_set)
        self.intervened = to_set(intervened)

    def __str__(self):
        marginal_str = ",".join(self.active_variables)
        if not self.intervened and not self.cond_set:
            return f'P({marginal_str})'
        else:
            do_str = f'do({",".join(self.intervened)})' if self.intervened else ''
            cond_str = ",".join(self.cond_set) if self.cond_set else ''
            sep = ',' if (do_str and cond_str) else ''
            return f'P({marginal_str}|{do_str}{sep}{cond_str})'

    def get_conditional(self, marginal=None, cond_set=None):
        super().get_conditional(marginal, cond_set)
        marginal, cond_set = to_set(marginal), to_set(cond_set)

        if len(marginal) == 0:
            active_variables = self.active_variables - cond_set
        else:
            assert marginal <= self.active_variables
            active_variables = marginal

        return ProbabilityTerm(active_variables, cond_set=self.cond_set | cond_set, intervened=self.intervened)

    def marginalize_out(self, marginalized_variables: set):
        return super().marginalize_out(marginalized_variables)


class Product(IDFormula):
    def __init__(self, terms: List[IDFormula]):
        super().__init__()
        self.terms = terms
        self.active_variables = set.union(*(term.active_variables for term in terms))

    def __str__(self):
        return '*'.join((str(term) for term in self.terms))

    def get_conditional(self, marginal=None, cond_set=None):
        super().get_conditional(marginal, cond_set)
        marginal, cond_set = to_set(marginal), to_set(cond_set)

        if len(cond_set) == 0:
            return MarginalDistribution(self, marginal)
        else:
            if len(marginal) == 0:
                marginal = self.active_variables - cond_set

            prod = Product([term.get_conditional(cond_set=cond_set) for term in self.terms])
            if (marginal | cond_set) == self.active_variables:
                return prod
            else:
                return MarginalDistribution(prod, marginal)

    def marginalize_out(self, marginalized_variables: set):
        return super().marginalize_out(marginalized_variables)


class MarginalDistribution(IDFormula):
    def __init__(self, distribution: IDFormula, marginal_variables=None, marginalized_variables=None):
        super().__init__()
        self.distribution = distribution

        assert xor(marginal_variables is not None, marginalized_variables is not None)
        if marginal_variables is not None:
            assert marginal_variables <= distribution.active_variables
            self.active_variables = marginal_variables
            self.marginalized_variables = distribution.active_variables - marginal_variables
        else:
            assert marginalized_variables <= distribution.active_variables
            self.active_variables = distribution.active_variables - marginalized_variables
            self.marginalized_variables = marginalized_variables

    def __str__(self):
        if len(self.marginalized_variables) != 0:
            return f'\\sum_{self.marginalized_variables} {self.distribution}'
        else:
            return f'{self.distribution}'

    def get_conditional(self, marginal=None, cond_set=None):
        super().get_conditional(marginal, cond_set)
        marginal, cond_set = to_set(marginal), to_set(cond_set)

        if len(cond_set) == 0:
            return MarginalDistribution(self.distribution, marginal_variables=marginal)
        if len(marginal) == 0:
            marginal = self.active_variables - cond_set

        conditional = self.distribution.get_conditional(cond_set=cond_set)
        marginal_variables = self.active_variables if marginal is None else marginal
        return MarginalDistribution(conditional, marginal_variables=marginal_variables)

    def marginalize_out(self, marginalized_variables: set):
        return super().marginalize_out(marginalized_variables)


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

    def ancestors_of(self, node: Union[Node, Set[Node]], include_argument=True) -> Set[Node]:
        ancestors = set() if not include_argument else to_set(node)
        self._add_ancestors(ancestors, to_set(node))
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

    def topological_sort(self) -> List[Node]:
        any_visited = {node: False for node in self._nodes}
        curr_path_visited = {node: False for node in self._nodes}
        curr_path = []
        stack = []
        for node in self._nodes:
            if not any_visited[node]:
                self._mark_children_visited(node, any_visited, curr_path_visited, curr_path, stack)
        return list(reversed(stack))

    def _mark_children_visited(self, node, any_visited, curr_path_visited, curr_path, stack):
        any_visited[node] = True
        curr_path_visited[node] = True
        curr_path.append(node)
        for child in self._children[node]:
            if not any_visited[child]:
                self._mark_children_visited(child, any_visited, curr_path_visited, curr_path, stack)
            elif curr_path_visited[child]:
                cycle = curr_path + [child]
                raise Exception(f"cycle! {cycle}")
        curr_path.pop()
        curr_path_visited[node] = False
        stack.append(node)

    def general_identification(self, y: set, x: Optional[Set[Node]], available_experiments: Optional[Set[frozenset]]):
        x, y = to_set(x), to_set(y)

        # LINE 2: return matching experiment if one exists
        matching_experiment = next((e for e in available_experiments if x == (e & self._nodes)), None)
        if matching_experiment:
            return ProbabilityTerm(y, intervened=matching_experiment)

        # LINE 3: simplify graph to only the ancestors of Y
        ancestors_y = self.ancestors_of(y)
        if ancestors_y != self._nodes:
            new_graph = self.induced_subgraph(ancestors_y)
            return new_graph.general_identification(y, x & ancestors_y, available_experiments)

        # LINE 4: fix values of ancestors of Y
        x_mutilated_graph = self.mutilated_graph(incoming_mutilated=x)
        ancestors_y_no_x = x_mutilated_graph.ancestors_of(y)
        w = self._nodes - x - ancestors_y_no_x
        if len(w) != 0:
            return self.general_identification(y, x | w, available_experiments)

        # LINE 5: get c-components of graph without X
        g_minus_x = self.induced_subgraph(removed_nodes=x)
        s_components = g_minus_x.c_components()

        # LINE 6: factorize into c-components
        if len(s_components) > 1:
            prod = Product([
                self.general_identification(s, self._nodes - s, available_experiments)
                for s in s_components
            ])
            return prod.get_conditional(marginal=y | x)

        # LINE 7: identify P_x using P_z as base distribution, if possible
        for z in available_experiments:
            if (z & self._nodes) <= x:
                g_no_zx = self.induced_subgraph(removed_nodes=z|x)
                distribution = ProbabilityTerm(self._nodes - z - x, intervened=(z - self._nodes) | (x & z))  # TODO right distribution?
                sub_id = g_no_zx.sub_identification(y, x - z, distribution)
                if sub_id is not None:
                    return sub_id

        # LINE 8
        raise NonIdentifiabilityError

    def sub_identification(self, y, x, distribution):
        # LINE 11: if no intervention, marginalize
        if len(x) == 0:
            return distribution.marginalize_out(self._nodes - y)

        # LINE 12: reduce to only ancestors of Y
        ancestors_y = self.ancestors_of(y)
        if ancestors_y != self._nodes:
            new_graph = self.induced_subgraph(ancestors_y)
            distribution = distribution.marginalize_out(self._nodes - ancestors_y)
            return new_graph.sub_identification(y, x & ancestors_y, distribution)

        # LINE 13: identification not possible if graph is a C-component
        c_components = self.c_components()
        if len(c_components) == 1:
            return None

        # LINE 10
        g_no_x = self.induced_subgraph(removed_nodes=x)
        s_components = g_no_x.c_components()
        assert len(s_components) == 1
        s = s_components[0]

        ordering = self.topological_sort()
        node2ix = {node: ix for ix, node in enumerate(ordering)}
        # LINE 14: if C-component of graph without X is a C-component,
        for c in c_components:
            if s == c:
                prod = Product([
                    distribution.get_conditional(marginal=v, cond_set=ordering[:node2ix[v]])
                    for v in y
                ])
                return prod.marginalize_out(s-y)

        # LINE 15
        for s_ in c_components:
            if s < s_:
                new_graph = self.induced_subgraph(s_)
                prod = Product([
                    distribution.get_conditional(marginal=v, cond_set=ordering[:node2ix[v]])
                    for v in s_
                ])
                return new_graph.sub_identification(y, x & s_, prod)

