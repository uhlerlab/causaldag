from causaldag.classes.dag import DAG
import numpy as np
from causaldag.utils import core_utils


class GaussDAG(DAG):
    def __init__(self, nodes=None, arcs=None, weight_mat=None, variances=None):
        if weight_mat is None:
            super().__init__(nodes, arcs)
            self._node_list = list(nodes)
            self._weight_mat = np.zeros((len(nodes), len(nodes)))
        else:
            self._weight_mat = weight_mat.copy()
            nnodes = weight_mat.shape[0]
            if nodes is None:
                self._node_list = list(range(nnodes))
            if variances is None:
                self._variances = np.ones(nnodes)

            arcs = set()
            for edge, val in np.ndenumerate(weight_mat):
                if val != 0:
                    arcs.add(edge)
            super().__init__(self._node_list, arcs)

        self._node2ix = core_utils.ix_map_from_list(self._node_list)
        self._precision = None
        self._covariance = None

    def set_arc_weight(self, i, j, val):
        self._weight_mat[i, j] = val
        if val == 0 and (i, j) in self._arcs:
            super().remove_arc(i, j)
        if val != 0 and (i, j) not in self._arcs:
            super().add_arc(i, j)

    def set_node_variance(self, i, var):
        self._variances[i] = var

    @property
    def weight_mat(self):
        return self._weight_mat.copy()

    @property
    def variances(self):
        return self._variances.copy()

    @property
    def precision(self):
        self._ensure_precision()
        return self._precision.copy()

    @property
    def covariance(self):
        self._ensure_covariance()
        return self._covariance.copy()

    def add_arc(self, i, j):
        self.set_arc_weight(i, j, 1)

    def remove_arc(self, i, j, ignore_error=False):
        self.set_arc_weight(i, j, 0)

    def add_node(self, node):
        self._node_list.append(node)
        self._weight_mat = np.zeros((len(self._node_list), len(self._node_list)))
        self._weight_mat[:-1, :-1] = None

    def remove_node(self, node, ignore_error=False):
        del self._node_list[self._node2ix[node]]
        self._weight_mat = self._weight_mat[np.ix_(self._node_list, self._node_list)]
        super().remove_node(node)

    def add_arcs_from(self, arcs):
        pass

    def add_nodes_from(self, nodes):
        pass

    def reverse_arc(self, i, j, ignore_error=False):
        pass

    def _ensure_precision(self):
        if self._precision is None:
            id_ = np.eye(len(self._nodes))
            a = self._weight_mat
            if (self._variances == 1).all():
                self._precision = (id_ - a) @ (id_ - a).T
            else:
                self._precision = (id_ - a) @ np.diag(self._variances ** -1) @ (id_ - a).T

    def _ensure_covariance(self):
        if self._covariance is None:
            id_ = np.eye(len(self._nodes))
            a = self._weight_mat
            id_min_a_inv = np.linalg.inv(id_ - a)
            if (self._variances == 1).all():
                self._covariance = id_min_a_inv.T @ id_min_a_inv
            else:
                self._covariance = id_min_a_inv.T @ np.diag(self._variances ** -1) @ id_min_a_inv

    def sample(self, nsamples):
        samples = np.zeros((nsamples, len(self._nodes)))
        noise = np.zeros((nsamples, len(self._nodes)))
        for ix, var in enumerate(self._variances):
            noise[:,ix] = np.random.normal(scale=var, size=nsamples)
        t = self.topological_sort()
        for node in t:
            ix = self._node2ix[node]
            parents = self._parents[node]
            if len(parents) != 0:
                parent_ixs = [self._node2ix[p] for p in self._parents[node]]
                parent_vals = samples[:, parent_ixs]
                samples[:, ix] = np.sum(parent_vals * self._weight_mat[parent_ixs, node], axis=1) + noise[:, ix]
            else:
                samples[:, ix] = noise[:, ix]
        return samples


if __name__ == '__main__':
    B = np.zeros((3, 3))
    B[0, 1] = 1
    B[0, 2] = -1
    B[1, 2] = 4
    gdag = GaussDAG(weight_mat=B)
    s = gdag.sample(1000)
    # print(gdag.arcs)
    print(s.T @ s / 1000)
    print(gdag.covariance)




