from typing import Union, List, Dict


def dsep_invariance_test(
        suffstat: Dict,
        context,
        i,
        cond_set: Union[List[int], int]=None
):
    dag = suffstat['dag']
    intervened_nodes = suffstat[context]
    return dict(reject=not dag.is_invariant(i, intervened_nodes, cond_set=cond_set))
