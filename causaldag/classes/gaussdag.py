# Author: Chandler Squires
"""
Base class for DAGs representing Gaussian distributions (i.e. linear SEMs with Gaussian noise).
"""

from causaldag.classes.dag import DAG
import numpy as np
from causaldag.utils import core_utils
from typing import Any, Dict, Union, Set, Tuple, List, FrozenSet, NewType
from dataclasses import dataclass
from scipy.linalg import ldl
import operator as op
from scipy.stats import norm


class InterventionalDistribution:
    def sample(self, size: int) -> np.array:
        raise NotImplementedError

    def pdf(self, vals: np.array) -> float:
        raise NotImplementedError


Intervention = NewType('Intervention', Dict[Any, InterventionalDistribution])


@dataclass
class GaussIntervention(InterventionalDistribution):
    mean: float = 0
    variance: float = 1

    def sample(self, size: int) -> np.array:
        samples = np.random.normal(loc=self.mean, scale=self.variance, size=size)
        return samples

    def pdf(self, vals: np.array) -> float:
        return norm.pdf(vals, loc=self.mean, scale=self.variance)


@dataclass
class BinaryIntervention(InterventionalDistribution):
    intervention1: InterventionalDistribution
    intervention2: InterventionalDistribution
    p: float = .5

    def sample(self, size: int) -> np.array:
        choices = np.random.binomial(1, self.p, size=size)
        ixs_iv1 = np.where(choices == 1)[0]
        ixs_iv2 = np.where(choices == 0)[0]
        samples = np.zeros(size)
        samples[ixs_iv1] = self.intervention1.sample(len(ixs_iv1))
        samples[ixs_iv2] = self.intervention2.sample(len(ixs_iv2))
        return samples

    def pdf(self, vals: np.array) -> float:
        return self.p * self.intervention1.pdf(vals) + (1 - self.p) * self.intervention2.pdf(vals)


@dataclass
class MultinomialIntervention(InterventionalDistribution):
    pvals: List[float]
    interventions: List[InterventionalDistribution]

    def sample(self, size: int) -> np.array:
        choices = np.random.choice(list(range(len(self.interventions))), size=size, p=self.pvals)
        samples = np.zeros(size)
        for ix, iv in enumerate(self.interventions):
            ixs_iv = np.where(choices == ix)[0]
            samples[ixs_iv] = iv.sample(len(ixs_iv))
        return samples

    def pdf(self, vals: np.array) -> float:
        raise NotImplementedError


@dataclass
class ConstantIntervention(InterventionalDistribution):
    val: float

    def sample(self, size: int) -> np.array:
        return np.ones(size) * self.val

    def pdf(self, vals: np.array) -> float:
        return (vals == self.val).astype(bool)


class GaussDAG(DAG):
    def __init__(self, nodes: List, arcs: Union[Set[Tuple[Any, Any]], Dict[Tuple[Any, Any], float]], means=None,
                 variances=None):
        arcs_set = arcs if isinstance(arcs, set) else set(arcs.keys())
        super().__init__(set(nodes), arcs_set)

        self._node_list = nodes
        self._node2ix = core_utils.ix_map_from_list(self._node_list)

        self._weight_mat = np.zeros((len(nodes), len(nodes)))
        for node1, node2 in arcs:
            w = arcs[(node1, node2)] if isinstance(arcs, dict) else 1
            self._weight_mat[self._node2ix[node1], self._node2ix[node2]] = w

        self._variances = np.ones(len(nodes)) if variances is None else np.array(variances, dtype=float)
        self._means = np.zeros((len(nodes))) if means is None else np.array(means)

        self._precision = None
        self._covariance = None

    @classmethod
    def from_amat(cls, weight_mat, nodes=None, means=None, variances=None):
        nodes = nodes if nodes is not None else list(range(weight_mat.shape[0]))
        arcs = {(i, j): w for (i, j), w in np.ndenumerate(weight_mat) if w != 0}
        return cls(nodes=nodes, arcs=arcs, means=means, variances=variances)

    @classmethod
    def from_precision(cls, precision_mat, node_order):
        from sksparse.cholmod import cholesky
        p = precision_mat.shape[0]

        # === permute precision matrix into  correct order for LDL
        precision_mat = precision_mat.copy()
        precision_mat = precision_mat[node_order]
        precision_mat = precision_mat[:, node_order]

        # === perform ldl decomposition and correct for floating point errors
        u, d, perm_ = ldl(precision_mat, lower=False)
        u[np.isclose(u, 0)] = 0

        # === permute back
        inv_node_order = [i for i, j in sorted(enumerate(node_order), key=op.itemgetter(1))]
        u = u.copy()
        u = u[inv_node_order]
        u = u[:, inv_node_order]
        d = d.copy()
        d = d[inv_node_order]
        d = d[:, inv_node_order]

        amat = np.eye(p) - u
        variances = np.diag(d) ** -1

        print(u)
        print(amat)
        print(d)
        print(variances)

        # adj_mat[np.isclose(adj_mat, 0)] = 0
        return GaussDAG.from_amat(amat, variances=variances)

    def set_arc_weight(self, i, j, val):
        self._weight_mat[self._node2ix[i], self._node2ix[j]] = val
        if val == 0 and (i, j) in self._arcs:
            super().remove_arc(i, j)
        if val != 0 and (i, j) not in self._arcs:
            super().add_arc(i, j)

    def set_node_variance(self, i, var):
        self._variances[i] = var

    @property
    def nodes(self):
        return list(self._nodes)

    @property
    def arc_weights(self):
        return {(i, j): self._weight_mat[i, j] for i, j in self._arcs}

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
        raise NotImplementedError
        pass

    def add_nodes_from(self, nodes):
        raise NotImplementedError
        pass

    def reverse_arc(self, i, j, ignore_error=False):
        raise NotImplementedError
        pass

    def vstructs(self):
        return super().vstructs()

    def reversible_arcs(self):
        return super().reversible_arcs()

    def topological_sort(self):
        return super().topological_sort()

    def shd(self, other):
        return super().shd(other)

    def downstream(self, node):
        return super().downstream(node)

    def upstream(self, node):
        return super().upstream(node)

    def incident_arcs(self, node):
        return super().incident_arcs(node)

    def incoming_arcs(self, node):
        return super().incoming_arcs(node)

    def outgoing_arcs(self, node):
        return super().outgoing_arcs(node)

    def outdegree(self, node):
        return super().outdegree(node)

    def indegree(self, node):
        return super().indegree(node)

    def save_gml(self, filename):
        raise NotImplementedError

    def to_amat(self):
        return self.weight_mat

    def cpdag(self):
        raise NotImplementedError

    def optimal_intervention_greedy(self, cpdag=None):
        return super().optimal_intervention_greedy(cpdag=cpdag)

    def backdoor(self, i, j):
        return super().backdoor(i, j)

    def frontdoor(self, i, j):
        return super().frontdoor(i, j)

    def dsep(self, i, j, c=None):
        return super().dsep(i, j, c=c)

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
                self._covariance = id_min_a_inv.T @ np.diag(self._variances) @ id_min_a_inv

    def sample(self, nsamples: int = 1) -> np.array:
        samples = np.zeros((nsamples, len(self._nodes)))
        noise = np.zeros((nsamples, len(self._nodes)))
        for ix, var in enumerate(self._variances):
            noise[:, ix] = np.random.normal(scale=var, size=nsamples)
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

    def sample_interventional(self, interventions: Intervention, nsamples: int = 1) -> np.array:
        samples = np.zeros((nsamples, len(self._nodes)))
        noise = np.zeros((nsamples, len(self._nodes)))

        for ix, (node, mean, var) in enumerate(zip(self._node_list, self._means, self._variances)):
            iv = interventions.get(node)
            if iv is not None:
                noise[:, ix] = iv.sample(nsamples)
            else:
                noise[:, ix] = np.random.normal(loc=mean, scale=var, size=nsamples)

        t = self.topological_sort()
        for node in t:
            ix = self._node2ix[node]
            parents = self._parents[node]
            if node not in interventions and len(parents) != 0:
                parent_ixs = [self._node2ix[p] for p in self._parents[node]]
                parent_vals = samples[:, parent_ixs]
                samples[:, ix] = np.sum(parent_vals * self._weight_mat[parent_ixs, node], axis=1) + noise[:, ix]
            else:
                samples[:, ix] = noise[:, ix]

        return samples

    def logpdf(self, samples: np.array, interventions: Intervention = None) -> np.array:
        sorted_nodes = self.topological_sort()
        nsamples = samples.shape[0]
        log_probs = np.zeros(nsamples)

        if interventions is None:
            for node in sorted_nodes:
                node_ix = self._node2ix[node]
                parent_ixs = [self._node2ix[p] for p in self._parents[node]]
                if len(parent_ixs) != 0:
                    parent_vals = samples[:, parent_ixs]
                    log_probs += norm.logpdf(samples[:, node_ix] - (parent_vals * self._weight_mat[parent_ixs, node]).sum(axis=1))
                else:
                    log_probs += norm.logpdf(samples[:, node_ix])
        else:
            for node in sorted_nodes:
                node_ix = self._node2ix[node]
                iv = interventions.get(node)
                if iv is not None:
                    log_probs += np.log(iv.pdf(samples[:, node_ix]))
                else:
                    parent_ixs = [self._node2ix[p] for p in self._parents[node]]
                    parent_vals = samples[:, parent_ixs]
                    log_probs += norm.logpdf(samples[:, node_ix] - (parent_vals * self._weight_mat[parent_ixs, node]).sum(axis=1))

        return log_probs


if __name__ == '__main__':
    iv = MultinomialIntervention(
        interventions=[
            ConstantIntervention(val=-1).sample,
            ConstantIntervention(val=1).sample,
            GaussIntervention(mean=2, variance=1).sample,
        ],
        pvals=[.4, .4, .2]
    )

    iv = BinaryIntervention(
        intervention1=ConstantIntervention(val=-1).sample,
        intervention2=ConstantIntervention(val=1).sample
    )
    B = np.zeros((3, 3))
    B[0, 1] = 1
    B[0, 2] = -1
    B[1, 2] = 4
    gdag = GaussDAG.from_amat(B, means=[0, 0, 0], variances=[1, 1, 1])
    s = gdag.sample(1000)
    # print(gdag.arcs)
    print(s.T @ s / 1000)
    print(gdag.covariance)
    s2 = gdag.sample_interventional({1: iv}, 1000)
    print(s2.T @ s2 / 1000)

    import matplotlib.pyplot as plt

    plt.clf()
    plt.ion()
    plt.scatter(s2[:,1], s2[:,2])
    plt.show()
