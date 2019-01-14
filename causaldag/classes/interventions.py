from typing import NewType, Dict, Any, List, Union, Optional
import numpy as np
from dataclasses import dataclass
from scipy.stats import norm


class PerfectInterventionalDistribution:
    def sample(self, size: int) -> np.ndarray:
        raise NotImplementedError

    def pdf(self, vals: np.ndarray) -> float:
        raise NotImplementedError


class SoftInterventionalDistribution:
    def sample(self, parent_values: np.ndarray, dag, node) -> np.ndarray:
        raise NotImplementedError

    def pdf(self, vals: np.ndarray, parent_values: np.ndarray, dag, node) -> float:
        raise NotImplementedError


InterventionalDistribution = NewType('InterventionalDistribution', Union[PerfectInterventionalDistribution, SoftInterventionalDistribution])
PerfectIntervention = NewType('Intervention', Dict[Any, PerfectInterventionalDistribution])
SoftIntervention = NewType('Intervention', Dict[Any, SoftInterventionalDistribution])


@dataclass
class ScalingIntervention(SoftInterventionalDistribution):
    factor: float = 1

    def sample(self, parent_values: Optional[np.ndarray], dag, node) -> np.ndarray:
        nsamples, nparents = parent_values.shape
        node_ix = dag._node2ix[node]
        noise = np.random.normal(scale=dag._variances[node_ix], size=nsamples)
        parent_ixs = [dag._node2ix[p] for p in dag._parents[node]]
        if nparents != 0:
            return np.sum(parent_values * dag._weight_mat[parent_ixs, node]*self.factor, axis=1) + noise
        else:
            return noise

    def pdf(self, vals: np.ndarray, parent_values: np.ndarray, dag, node) -> float:
        pass


@dataclass
class GaussIntervention(PerfectInterventionalDistribution):
    mean: float = 0
    variance: float = 1

    def sample(self, size: int) -> np.ndarray:
        samples = np.random.normal(loc=self.mean, scale=self.variance**.5, size=size)
        return samples

    def pdf(self, vals: np.ndarray) -> float:
        return norm.pdf(vals, loc=self.mean, scale=self.variance**.5)

    def logpdf(self, vals: np.ndarray) -> float:
        return norm.logpdf(vals, loc=self.mean, scale=self.variance**.5)


@dataclass
class BinaryIntervention(PerfectInterventionalDistribution):
    intervention1: PerfectInterventionalDistribution
    intervention2: PerfectInterventionalDistribution
    p: float = .5

    def sample(self, size: int) -> np.ndarray:
        choices = np.random.binomial(1, self.p, size=size)
        ixs_iv1 = np.where(choices == 1)[0]
        ixs_iv2 = np.where(choices == 0)[0]
        samples = np.zeros(size)
        samples[ixs_iv1] = self.intervention1.sample(len(ixs_iv1))
        samples[ixs_iv2] = self.intervention2.sample(len(ixs_iv2))
        return samples

    def pdf(self, vals: np.ndarray) -> float:
        return self.p * self.intervention1.pdf(vals) + (1 - self.p) * self.intervention2.pdf(vals)


@dataclass
class MultinomialIntervention(PerfectInterventionalDistribution):
    pvals: List[float]
    interventions: List[PerfectInterventionalDistribution]

    def sample(self, size: int) -> np.ndarray:
        choices = np.random.choice(list(range(len(self.interventions))), size=size, p=self.pvals)
        samples = np.zeros(size)
        for ix, iv in enumerate(self.interventions):
            ixs_iv = np.where(choices == ix)[0]
            samples[ixs_iv] = iv.sample(len(ixs_iv))
        return samples

    def pdf(self, vals: np.ndarray) -> float:
        raise NotImplementedError


@dataclass
class ConstantIntervention(PerfectInterventionalDistribution):
    val: float

    def sample(self, size: int) -> np.ndarray:
        return np.ones(size) * self.val

    def pdf(self, vals: np.ndarray) -> float:
        return np.all(vals == self.val).astype(float)

