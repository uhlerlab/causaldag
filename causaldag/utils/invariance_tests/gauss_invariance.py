import numpy as np
from typing import Union, List, Optional
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
from scipy.stats import f as fdist
from causaldag.utils.core_utils import to_list

lr = LinearRegression()


def gauss_invariance_test(
        samples1: np.ndarray,
        samples2: np.ndarray,
        i: int,
        cond_set: Optional[Union[List[int], int]]=None,
        alpha: float=0.05,
):
    cond_set = to_list(cond_set)
    lr.fit(samples1[:, cond_set], samples1[:, i])
    residuals1 = samples1[:, i] - samples1[:, cond_set] @ lr.coef_
    lr.fit(samples2[:, cond_set], samples2[:, i])
    residuals2 = samples2[:, i] - samples2[:, cond_set] @ lr.coef_

    ttest_results = ttest_ind(residuals1, residuals2, equal_var=False)
    ftest_stat = np.var(residuals1)/np.var(residuals2)
    f_pvalue = 1 - fdist.cdf(ftest_stat, samples1.shape[0]-1, samples2.shape[0]-1)

    return dict(
        ttest_stat=ttest_results.statistic,
        ftest_stat=ftest_stat,
        reject=f_pvalue<alpha/2 or ttest_results.pvalue<alpha/2
    )


