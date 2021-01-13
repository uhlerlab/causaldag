

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
    from causaldag.rand import rand_weights, directed_erdos
    from causaldag.utils.ci_tests import partial_correlation_suffstat
    from causaldag.utils.scores.gaussian_bic_score import local_gaussian_bic_score

    d = directed_erdos(10, .5)
    g = rand_weights(d)
    samples = g.sample(100)
    suffstat = partial_correlation_suffstat(samples)
    scorer = MemoizedDecomposableScore(local_gaussian_bic_score, suffstat)
    score = scorer.get_score(d)


