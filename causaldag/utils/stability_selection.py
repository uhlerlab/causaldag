from sklearn.utils.random import sample_without_replacement
import numpy as np
from joblib import Parallel, delayed
from frozendict import frozendict
from collections import Counter
import ipdb
from sklearn.utils import safe_mask
import random


def bootstrap_generator(n_bootstrap_iterations, sample_fraction, nsamples, random_state=None):
    """Generates bootstrap samples from dataset."""
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    n_subsamples = np.floor(sample_fraction * nsamples).astype(int)
    for _ in range(n_bootstrap_iterations):
        subsample = sample_without_replacement(nsamples, n_subsamples)
        yield subsample


def stability_selection(
        method
):
    def wrapped_method(
            *args,
            parameter_grid: list,
            n_jobs: int = 1,
            n_bootstrap_iterations: int = 50,
            random_state: int = 0,
            sample_fraction: float = 0.7,
            verbose: bool = False,
            bootstrap_threshold: float = 0.5,
            **kwargs
    ):
        parameters2results = dict()

        for parameters in parameter_grid:
            arg2subsamples_list = []
            for X in args:
                nsamples = X.shape[0]
                bootstrap_samples = list(bootstrap_generator(n_bootstrap_iterations, sample_fraction, nsamples, random_state=random_state))
                arg2subsamples_list.append(bootstrap_samples)

            bootstrap_results = Parallel(n_jobs, verbose=verbose)(
                delayed(method)(
                    *(arg[safe_mask(arg, subsample), :] for arg, subsample in zip(args, arg2subsamples)),
                    **kwargs,
                    **parameters
                ) for arg2subsamples in zip(*arg2subsamples_list)
            )

            parameters2results[frozendict(parameters)] = bootstrap_results

        stable_results = set()
        for param, results in parameters2results.items():
            counter = Counter()
            for result in results:
                counter.update(result[0])
            for item, count in counter.items():
                if count >= bootstrap_threshold*n_bootstrap_iterations:
                    stable_results.add(item)

        return stable_results, parameters2results

    return wrapped_method
