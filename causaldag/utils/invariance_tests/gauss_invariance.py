import numpy as np
from typing import Union, List, Optional
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
from scipy.stats import f as fdist
from causaldag.utils.core_utils import to_list

lr = LinearRegression()


def gauss_invariance_test(
        suffstat,
        context,
        i: int,
        cond_set: Optional[Union[List[int], int]]=None,
        alpha: float=0.05,
):
    cond_set = to_list(cond_set)
    obs_samples = suffstat['obs_samples']
    iv_samples = suffstat[context]

    if len(cond_set) != 0:
        lr.fit(obs_samples[:, cond_set], obs_samples[:, i])
        residuals1 = obs_samples[:, i] - obs_samples[:, cond_set] @ lr.coef_
        lr.fit(iv_samples[:, cond_set], iv_samples[:, i])
        residuals2 = iv_samples[:, i] - iv_samples[:, cond_set] @ lr.coef_
    else:
        residuals1 = obs_samples[:, i]
        residuals2 = iv_samples[:, i]

    ttest_results = ttest_ind(residuals1, residuals2, equal_var=False)
    ftest_stat = np.var(residuals1)/np.var(residuals2)
    f_pvalue = 1 - fdist.cdf(ftest_stat, obs_samples.shape[0]-1, iv_samples.shape[0]-1)

    return dict(
        ttest_stat=ttest_results.statistic,
        ftest_stat=ftest_stat,
        reject=f_pvalue<alpha/2 or ttest_results.pvalue<alpha/2
    )


