from causaldag.classes.pdag import PDAG
import itertools as itr


def ges(score, nnodes):
    nodes = set(range(nnodes))
    current_pdag = PDAG(nodes=nodes)
    valid_insertions = dict()

    # FORWARD PHASE
    # === insert X->Y.
    # Valid if:
    # - Subset T of nbrs(Y)-adj(X) forms a clique w/ NA(YX) = nbrs(Y)&adj(X)
    # - every semi-directed path from Y to X contains node in NA(YX)|T
    while True:
        for x, y in itr.combinations(nodes, r=2):
            if not current_pdag.has_edge_or_arc(x, y):
                NA_YX = current_pdag._undirected_neighbors[y] & current_pdag._neighbors[x]
                # get clique C in undirected neighbors of Y s.t. NA_YX in C
                # check there's no semidirected path
                # compute increment in score
                T0 = current_pdag._undirected_neighbors[y] - current_pdag._neighbors[x]
                # go through subsets of T0


    valid_deletions = dict()
    # BACKWARD PHASE
    # === remove X-Y or X->Y
    # Valid if:
    # - Subset H of nbrs(Y)&adj(X) st NA(YX)-H is a clique
    raise NotImplementedError


