import random
from math import exp
from tqdm import trange
import numpy as np
import sys
sys.path.insert(1, "C:/Users/skarn/OneDrive/Documents/MIT/year_3/SuperUROP/causaldag")

from causaldag import DAG
from causaldag.rand.graphs import directed_erdos, rand_weights
from causaldag import permutation2dag
from causaldag.utils.ci_tests import MemoizedCI_Tester, partial_correlation_test, partial_correlation_suffstat, partial_monte_carlo_correlation_suffstat
from causaldag.utils.samplers.proposals.transposition_proposers import adjacent_transposition_proposer
from causaldag.utils.scores import MemoizedDecomposableScore
from causaldag.utils.scores.gaussian_bic_score import local_gaussian_bic_score
from causaldag.utils.scores.gaussian_bge_score import local_gaussian_bge_score
from causaldag.utils.scores.gaussian_ibge_score import local_bayesian_regression_bge_score

def minimal_imap_mcmc(
        initial_perm,
        initial_dag,
        ci_tester,
        proposer,
        scorer,
        burn=1000,
        num_steps=10000,
        thin=1,
        progress=False,
        verbose=False
):
    """
    Get DAG samples from the approximate posterior over minimal IMAPs.

    Parameters
    ----------
    initial_perm:
        TODO
    initial_dag:
        TODO
    ci_tester:
        A conditional independence tester, which has a method is_ci taking two elements i and j, and a conditioning set
        C, that returns True/False.
    proposer:
        A function that proposes new permutations from the current permutation.
    scorer:
        A function that evaluates the log-likelihood of a given DAG.
    burn:
        Number of burn-in steps.
    num_steps:
        Total number of steps to run the Markov chain.
    thin:
        The thinning rate, i.e., how many steps between taking samples.

    Returns
    -------
    List[DAG]
        sampled DAGs
    """
    current_perm, current_dag = initial_perm, initial_dag
    current_score = scorer.get_score(current_dag)

    samples = []
    r = trange if progress else range
    for step in r(num_steps):
        if step >= burn and (step - burn) % thin == 0:
            samples.append((current_dag, current_perm))
        proposal_dag, proposal_perm = proposer(current_dag, current_perm, ci_tester)
        proposal_score = scorer.get_score(proposal_dag)

        accept = proposal_score > current_score or random.random() < exp(proposal_score - current_score)
        if verbose: print(proposal_perm, exp(proposal_score - current_score))
        if accept:
            current_dag, current_perm, current_score = proposal_dag, proposal_perm, proposal_score

    return samples

def collect_stats_mcmc(
        true_dag,
        initial_perm,
        initial_dag,
        ci_tester,
        proposer,
        scorer,
        num_steps=100,
        burn=1,
        progress=False,
        verbose=True):
    
    samples = minimal_imap_mcmc(
        initial_perm,
        initial_dag,
        ci_tester,
        proposer,
        scorer,
        num_steps=100,
        burn=1
    )

    desired_arcs = true_dag.arcs
    total_samples = len(samples)
    correct_samples = np.sum([samples[i][0].arcs == desired_arcs for i in range(total_samples)])
    stats = {'samples': samples, 'fraction_correct': correct_samples/total_samples}
    return stats

if __name__ == '__main__':
    d = DAG(arcs={(0, 1), (2, 1)})
    # d = DAG(arcs={(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)})
    g = rand_weights(d)
    samples = g.sample(1000)
    suffstat = partial_correlation_suffstat(samples)
    suffstat['samples'] = samples
    initial_perm = [1, 2, 0]
    ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=.05)
    initial_dag = permutation2dag(initial_perm, ci_tester)
    # scorer = MemoizedDecomposableScore(local_gaussian_bge_score, suffstat)
    scorer_bge = MemoizedDecomposableScore(local_gaussian_bge_score, suffstat)
    num_meta_iterations = 30
    stats_bges = [collect_stats_mcmc(
        d,
        initial_perm,
        initial_dag,
        ci_tester,
        adjacent_transposition_proposer,
        scorer_bge,
        num_steps=100,
        burn=1,
        progress=False,
        verbose=False
    ) for i in range(num_meta_iterations)]

    bge_fraction_correct = np.mean([stats_bge["fraction_correct"] for stats_bge in stats_bges])
    print("BGe fraction correct:   ", bge_fraction_correct)

    scorer_ibge = MemoizedDecomposableScore(local_bayesian_regression_bge_score, suffstat)

    stats_ibges = [collect_stats_mcmc(
        d,
        initial_perm,
        initial_dag,
        ci_tester,
        adjacent_transposition_proposer,
        scorer_ibge,
        num_steps=100,
        burn=1,
        progress=False,
        verbose=False
    ) for i in range(num_meta_iterations)]

    ibge_fraction_correct = np.mean([stats_ibge["fraction_correct"] for stats_ibge in stats_ibges])
    print("I-BGe fraction correct: ", ibge_fraction_correct)


