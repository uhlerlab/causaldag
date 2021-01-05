import random
from math import exp
from tqdm import trange


def minimal_imap_mcmc(
        initial_perm,
        initial_dag,
        ci_tester,
        proposer,
        scorer,
        burn=1000,
        num_steps=10000,
        thin=1,
        progress=False
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
        if accept:
            current_dag, current_perm, current_score = proposal_dag, proposal_perm, proposal_score

    return samples

