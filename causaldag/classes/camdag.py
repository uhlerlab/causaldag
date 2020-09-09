from causaldag.classes import DAG
import numpy as np
from causaldag.utils import core_utils
from typing import Callable, Dict, Hashable, Optional
from tqdm import tqdm


class CamDAG(DAG):
    def __init__(self, nodes, arcs):
        super().__init__(set(nodes), arcs)
        self.mean_functions = dict()
        self.node_noises = dict()
        self._node_list = list(nodes)
        self._node2ix = core_utils.ix_map_from_list(self._node_list)

    def set_mean_function(self, node, mean_function: Callable[[np.ndarray, list], np.ndarray]):
        self.mean_functions[node] = mean_function

    def set_noise(self, node, noise_function: Callable[[], np.ndarray]):
        self.node_noises[node] = noise_function

    def sample(self, nsamples: int = 1, progress=False) -> np.array:
        samples = np.zeros((nsamples, len(self._nodes)))
        t = self.topological_sort()
        t = t if not progress else tqdm(t)

        for node in t:
            parents = list(self._parents[node])
            if len(parents) > 0:
                parent_ixs = [self._node2ix[p] for p in parents]
                means = self.mean_functions[node](samples[:, parent_ixs], parents)
            else:
                means = np.zeros(nsamples)
            noise = self.node_noises[node](size=nsamples)
            samples[:, self._node2ix[node]] = means + noise

        return samples

    def conditional_mean(self, cond_values: np.ndarray, cond_nodes: list, check_valid=False) -> np.ndarray:
        assert len(cond_nodes) == cond_values.shape[1]
        if check_valid:
            parents_of_cond = set.union(*(self.parents_of(node) for node in cond_nodes))
            assert parents_of_cond <= set(cond_nodes)
        t = self.topological_sort()
        remaining_nodes = [node for node in t if node not in set(cond_nodes)]

        nsamples = cond_values.shape[0]
        conditional_means = np.zeros((nsamples, self.nnodes))
        conditional_means[:, [self._node2ix[node] for node in cond_nodes]] = cond_values

        for node in remaining_nodes:
            parents = list(self._parents[node])
            parent_ixs = [self._node2ix[p] for p in parents]
            means = self.mean_functions[node](conditional_means[:, parent_ixs], parents)
            conditional_means[:, self._node2ix[node]] = means

        return conditional_means





