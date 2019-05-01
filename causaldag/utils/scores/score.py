

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






