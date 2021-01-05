import random
from causaldag import permutation2dag


def adjacent_transposition(dag, perm, ci_tester):
    ix = random.choice(list(range(len(perm)-1)))
    new_perm = perm.copy()
    new_perm[ix], new_perm[ix+1] = perm[ix+1], perm[ix]
    new_dag = permutation2dag(new_perm, ci_tester)
    return new_dag, new_perm
