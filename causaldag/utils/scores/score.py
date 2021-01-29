

class MemoizedDecomposableScore:
    def __init__(self, local_score, suffstat, **kwargs):
        self.local_score = local_score
        self.suffstat = suffstat
        self.kwargs = kwargs
        self.score_dict = dict()

    def get_local_score(self, node, parents):
        score = self.score_dict.get((node, frozenset(parents)))
        if score:
            return score
        score = self.local_score(node, parents, self.suffstat, **self.kwargs)
        self.score_dict[(node, frozenset(parents))] = score
        return score

    def get_score(self, dag):
        total_score = 0
        for node in dag.nodes:
            local_score = self.get_local_score(node, dag.parents_of(node))
            total_score += local_score
        return total_score


if __name__ == '__main__':
    import sys
    sys.path.insert(1, "C:/Users/skarn/OneDrive/Documents/MIT/year_3/SuperUROP/causaldag")
    import numpy as np
    from causaldag.rand import rand_weights, directed_erdos
    from causaldag import GaussIntervention
    from conditional_independence import partial_correlation_suffstat
    from causaldag.utils.scores.gaussian_ibge_score import local_bayesian_regression_bge_score, local_gaussian_interventional_bge_score
    from causaldag.utils.scores.gaussian_bic_score import local_gaussian_bic_score, local_gaussian_interventional_bic_score
    from causaldag.utils.suffstats.gaussian_interventional_suffstat import compute_gaussian_interventional_suffstat

    # d = directed_erdos(10, .5)
    # g = rand_weights(d)
    # samples = g.sample(100)
    # suffstat = partial_monte_carlo_correlation_suffstat(samples)
    # scorer = MemoizedDecomposableScore(local_gaussian_monte_carlo_bge_score, suffstat)
    # score = scorer.get_score(d)

    d = directed_erdos(10, .5)
    g = rand_weights(d)
    samples = g.sample(100)
    node = 5
    iv_samples = g.sample_interventional({node: GaussIntervention(0, 1)}, 100)
    data = {frozenset(): samples, frozenset({node}): iv_samples}
    suffstat = compute_gaussian_interventional_suffstat(data)
    # scorer = MemoizedDecomposableScore(local_gaussian_interventional_bic_score, suffstat)
    # score = scorer.get_score(d)
    for node in d.nodes:
        score = local_gaussian_interventional_bic_score(node, d.parents_of(node), suffstat, 0)
        print(node, score)
    
    print()

    all_samples = np.vstack((samples, iv_samples))
    suffstat = partial_correlation_suffstat(all_samples)
    for node in d.nodes:
        score = local_gaussian_bic_score(node, d.parents_of(node), suffstat, 0)
        print(node, score)

    print("***** I-BGe *****")

    suffstat = compute_gaussian_interventional_suffstat(data)
    # scorer = MemoizedDecomposableScore(local_gaussian_interventional_bic_score, suffstat)
    # score = scorer.get_score(d)
    for node in d.nodes:
        score = local_gaussian_interventional_bge_score(node, d.parents_of(node), suffstat)
        print(node, score)
    
    print()

    suffstat = partial_correlation_suffstat(all_samples)
    for node in d.nodes:
        score = local_bayesian_regression_bge_score(node, d.parents_of(node), suffstat)
        print(node, score)
