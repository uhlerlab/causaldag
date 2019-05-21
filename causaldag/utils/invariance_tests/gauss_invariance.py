import numpy as np
from typing import Union, List, Optional
from sklearn.linear_model import LinearRegression
from causaldag.utils.core_utils import to_list
from scipy.special import stdtr, ncfdtr


lr = LinearRegression()


def gauss_invariance_suffstat(
        obs_samples,
        context_samples_list
):
    """
    Helper function to compute the sufficient statistics for the gauss_invariance_test from data.

    Parameters
    ----------
    obs_samples:
        (n x p) matrix, where n is the number of samples and p is the number of variables.
    context_samples_list:
        list of (n x p) matrices, one for each context besides observational
    # invert:
    #     if True, compute the inverse correlation matrix, and normalize it into the partial correlation matrix. This
    #     will generally speed up the gauss_ci_test if large conditioning sets are used.

    Return
    ------
    dictionary of sufficient statistics
    """
    obs_suffstat = dict(samples=obs_samples, G=obs_samples.T@obs_samples)
    context_suffstats = []
    for context_samples in context_samples_list:
        context_suffstats.append(dict(samples=context_samples, G=context_samples.T@context_samples))

    return dict(obs=obs_suffstat, contexts=context_suffstats)


def gauss_invariance_test(
        suffstat,
        context,
        i: int,
        cond_set: Optional[Union[List[int], int]]=None,
        alpha: float=0.05,
        new=True
):
    """
    Test the null hypothesis that two Gaussian distributions are equal.

    Parameters
    ----------
    suffstat:
        dictionary containing:
        'obs' -- number of samples
            'G' -- Gram matrix
        'contexts'
    context:
        which context to test.
    i:
        position of marginal distribution.
    cond_set:
        positions of conditioning set in correlation matrix.
    alpha:
        Significance level.

    Return
    ------
    dictionary containing ttest_stat, ftest_stat, f_pvalue, t_pvalue, and reject.
    """
    cond_set = to_list(cond_set)
    obs_samples = suffstat['obs']['samples']
    iv_samples = suffstat['contexts'][context]['samples']
    n1 = obs_samples.shape[0]
    n2 = iv_samples.shape[0]

    if len(cond_set) != 0:
        if new:
            gram1 = suffstat['obs']['G'][np.ix_(cond_set, cond_set)]
            gram2 = suffstat['contexts'][context]['G'][np.ix_(cond_set, cond_set)]
            coefs1 = np.linalg.inv(gram1) @ obs_samples[:, cond_set].T @ obs_samples[:, i]
            coefs2 = np.linalg.inv(gram2) @ obs_samples[:, cond_set].T @ obs_samples[:, i]
        else:
            lr.fit(obs_samples[:, cond_set], obs_samples[:, i])
            coefs1 = lr.coef_
            lr.fit(iv_samples[:, cond_set], iv_samples[:, i])
            coefs2 = lr.coef_
        residuals1 = obs_samples[:, i] - obs_samples[:, cond_set] @ coefs1
        residuals2 = iv_samples[:, i] - iv_samples[:, cond_set] @ coefs2
    else:
        residuals1 = obs_samples[:, i]
        residuals2 = iv_samples[:, i]

    # means and variances of residuals
    var1, var2 = np.var(residuals1, ddof=1), np.var(residuals2, ddof=1)
    mean1, mean2 = np.mean(residuals1), np.mean(residuals2)

    # calculate statistic for T-test
    ttest_stat = (mean1 - mean2)/np.sqrt(var1/n1 + var2/n2)
    dof = (var1 / n1 + var2 / 2) ** 2 / (var1 ** 2 / (n1 ** 2 * (n1 - 1)) + var2 ** 2 / (n2 ** 2 * (n2 - 2)))
    t_pvalue = 2*stdtr(dof, -np.abs(ttest_stat))

    # calculate statistic for F-Test
    ftest_stat = var1/var2
    f_pvalue = ncfdtr(n1-1, n2-1, 0, ftest_stat)
    f_pvalue = 2*min(f_pvalue, 1-f_pvalue)

    return dict(
        ttest_stat=ttest_stat,
        ftest_stat=ftest_stat,
        f_pvalue=f_pvalue,
        t_pvalue=t_pvalue,
        reject=f_pvalue<alpha/2 or t_pvalue<alpha/2
    )


