from ...classes.dag import DAG
from ...classes.ancestral_graph import AncestralGraph
from typing import Union, List


def dsep_test(
        dag: DAG,
        i,
        j,
        cond_set: Union[List[int], int]=None
):
    return dict(reject=not dag.dsep(i, j, cond_set))


def msep_test(
        mag: AncestralGraph,
        i,
        j,
        cond_set: Union[List[int], int]=None
):
    return dict(reject=not mag.msep(i, j, cond_set))


def dsep_invariance_test(

):
    pass
