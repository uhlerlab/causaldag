from causaldag import DAG
from typing import Union, List


def dsep_invariance_test(
        dag: DAG,
        intervened_nodes,
        i,
        cond_set: Union[List[int], int]=None
):
    return dict(reject=not dag.is_invariant(i, intervened_nodes, cond_set=cond_set))
