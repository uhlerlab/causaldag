from ...classes.dag import DAG
from typing import Union, List


def dsep_test(
        dag: DAG,
        i,
        j,
        cond_set: Union[List[int], int]=None
):
    return dag.dsep(i, j, cond_set)


def dsep_invariance_test(

):
    pass
