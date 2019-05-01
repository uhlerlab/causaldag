from collections import defaultdict


class UndirectedGraph:
    def __init__(self, nodes, edges=set()):
        self._nodes = nodes.copy()
        self._edges = {tuple(sorted((i, j))) for i, j in edges}
        self._neighbors = defaultdict(set)
        self._degrees = defaultdict(int)
        for i, j in edges:
            self._neighbors[i].add(j)
            self._neighbors[j].add(i)
            self._degrees[i] += 1
            self._degrees[j] += 1

    def copy(self):
        return UndirectedGraph(self._nodes, self._edges)

    @property
    def degrees(self):
        return {node: self._degrees[node] for node in self._nodes}

    @property
    def neighbors(self):
        return {node: self._neighbors[node] for node in self._nodes}

    @property
    def edges(self):
        return self._edges.copy()

    @property
    def nodes(self):
        return self._nodes.copy()

    def has_edge(self, i, j):
        return tuple(sorted((i, j))) in self._edges

    def neighbors_of(self, node):
        return self._neighbors[node].copy()

    # === MUTATORS ===
    def add_edge(self, i, j):
        self._edges.add(tuple(sorted((i, j))))
        self._neighbors[i].add(j)
        self._neighbors[j].add(i)
        self._degrees[i] += 1
        self._degrees[j] += 1

    def delete_edge(self, i, j):
        self._edges.remove(tuple(sorted((i, j))))
        self._neighbors[i].remove(j)
        self._neighbors[j].remove(i)
        self._degrees[i] -= 1
        self._degrees[j] -= 1

    def delete_node(self, i):
        self._nodes.remove(i)
        for j in self._neighbors[i]:
            self._neighbors[j].remove(i)
            self._degrees[j] -= 1
            self._edges.remove(tuple(sorted((i, j))))
        self._neighbors.pop(i, None)
        self._degrees.pop(i, None)

