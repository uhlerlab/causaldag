from typing import Union, List, Optional
import numpy as np
from causaldag.utils.core_utils import to_list
from ._utils import combined_mat
from causaldag.utils.ci_tests.hsic import hsic_test


def hsic_invariance_test(
        suffstat,
        context,
        i: int,
        cond_set: Optional[Union[List[int], int]]=None,
        alpha: float=0.05
):
    cond_set = to_list(cond_set)
    obs_samples = suffstat['obs_samples']
    iv_samples = suffstat[context]

    mat = combined_mat(obs_samples, iv_samples, i, cond_set)
    return hsic_test(mat, 0, 1, list(range(2, 2+len(cond_set))), alpha=alpha)
