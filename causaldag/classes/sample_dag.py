from causaldag.classes import DAG
import numpy as np
from causaldag.utils import core_utils
from typing import Callable
from causaldag.classes.interventions import Intervention, SoftInterventionalDistribution, PerfectInterventionalDistribution
from tqdm import trange


class SampleDAG(DAG):
    def __init__(self, nodes, arcs):
        super().__init__(set(nodes), arcs)
        self.conditionals = dict()
        self._node_list = list(nodes)
        self._node2ix = core_utils.ix_map_from_list(self._node_list)

    def set_conditional(self, node, conditional_distribution: Callable[[np.ndarray], np.ndarray]):
        self.conditionals[node] = conditional_distribution

    def sample(self, nsamples: int = 1, progress=False) -> np.array:  # TODO: parallelize?
        samples = np.zeros((nsamples, len(self._nodes)))
        t = self.topological_sort()
        r = trange if progress else range
        for sample_num in r(nsamples):
            for node in t:
                node_ix = self._node2ix[node]
                parent_ixs = [self._node2ix[p] for p in self._parents[node]]
                samples[sample_num, node_ix] = self.conditionals[node](samples[sample_num, parent_ixs])
        return samples

    def sample_interventional(self, intervention: Intervention, nsamples: int = 1) -> np.ndarray:
        samples = np.zeros((nsamples, len(self._nodes)))

        t = self.topological_sort()
        for node in t:
            ix = self._node2ix[node]
            parent_ixs = [self._node2ix[p] for p in self._parents[node]]
            parent_vals = samples[:, parent_ixs]

            interventional_dist = intervention.get(node)
            if interventional_dist is not None:
                if isinstance(interventional_dist, SoftInterventionalDistribution):
                    samples[:, ix] = interventional_dist.sample(parent_vals, self, node)
                elif isinstance(interventional_dist, PerfectInterventionalDistribution):
                    samples[:, ix] = interventional_dist.sample(nsamples)
            else:
                for sample_num in range(nsamples):
                    samples[sample_num, ix] = self.conditionals[node](samples[sample_num, parent_ixs])

        return samples
