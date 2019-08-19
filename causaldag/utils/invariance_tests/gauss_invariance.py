import numpy as np
from typing import Union, List, Optional
from sklearn.linear_model import LinearRegression
from causaldag.utils.core_utils import to_list
from scipy.special import stdtr, ncfdtr
from numpy.linalg import inv


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
    obs_samples = np.hstack((obs_samples, np.ones([obs_samples.shape[0], 1])))
    obs_cov = np.cov(obs_samples, rowvar=False)
    obs_suffstat = dict(samples=obs_samples, G=obs_samples.T@obs_samples, S=obs_cov)
    context_suffstats = []
    for context_samples in context_samples_list:
        context_samples = np.hstack((context_samples, np.ones([context_samples.shape[0], 1])))
        context_cov = np.cov(context_samples, rowvar=False)
        context_suffstats.append(dict(samples=context_samples, G=context_samples.T@context_samples, S=context_cov))

    return dict(obs=obs_suffstat, contexts=context_suffstats)


def gauss_invariance_test(
        suffstat,
        context,
        i: int,
        cond_set: Optional[Union[List[int], int]]=None,
        alpha: float=0.05,
        new=True,
        zero_mean=False,
        zero_coeffs=False
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
    n1, p = obs_samples.shape
    n2 = iv_samples.shape[0]

    if len(cond_set) != 0:
        cond_ix = cond_set if zero_mean else [*cond_set, -1]
        gram1 = suffstat['obs']['G'][np.ix_(cond_ix, cond_ix)]
        gram2 = suffstat['contexts'][context]['G'][np.ix_(cond_ix, cond_ix)]
        coefs1 = np.linalg.inv(gram1) @ obs_samples[:, cond_ix].T @ obs_samples[:, i]
        coefs2 = np.linalg.inv(gram2) @ iv_samples[:, cond_ix].T @ iv_samples[:, i]

        # lr.fit(obs_samples[:, cond_set], obs_samples[:, i])
        # coefs1_ = lr.coef_
        # lr.fit(iv_samples[:, cond_set], iv_samples[:, i])
        # coefs2_ = lr.coef_

        residuals1 = obs_samples[:, i] - obs_samples[:, cond_ix] @ coefs1
        residuals2 = iv_samples[:, i] - iv_samples[:, cond_ix] @ coefs2
    else:
        gram1 = n1*np.ones([1, 1])
        gram2 = n2*np.ones([2,2])
        coefs1 = np.array([np.mean(obs_samples[:, i])]) if not zero_mean else 0
        coefs2 = np.array([np.mean(iv_samples[:, i])]) if not zero_mean else 0
        residuals1 = obs_samples[:, i] - coefs1
        residuals2 = iv_samples[:, i] - coefs2

    bias1 = coefs1[-1] if not zero_mean else 0
    bias2 = coefs2[-1] if not zero_mean else 0
    coefs1 = coefs1[:-1] if not zero_mean else coefs1
    coefs2 = coefs2[:-1] if not zero_mean else coefs2

    # means and variances of residuals
    var1, var2 = np.var(residuals1, ddof=len(coefs1)), np.var(residuals2, ddof=len(coefs2))

    # calculate regression coefficient invariance statistic
    print('===================')
    print(i, cond_set)
    if len(cond_set) != 0:
        gram1 = suffstat['obs']['G'][np.ix_(cond_set, cond_set)]
        gram2 = suffstat['contexts'][context]['G'][np.ix_(cond_set, cond_set)]
        rc_stat = (coefs1 - coefs2) @ inv(var1*inv(gram1) + var2*inv(gram2)) @ (coefs1 - coefs2).T / len(coefs1)
        rc_pvalue = ncfdtr(len(cond_set), n1 + n2 - len(cond_set), 0, rc_stat)
        rc_pvalue = 2*min(rc_pvalue, 1-rc_pvalue)

        # sigma = np.eye(n2) + iv_samples[:, cond_ix] @ inv(obs_samples[:, cond_ix].T @ obs_samples[:, cond_ix]) @ iv_samples[:, cond_ix].T
        # rc_stat = residuals2 @ inv(sigma) @ residuals2.T / var1 / n2
        # rc_pvalue = ncfdtr(n2, n1 - len(coefs1) - 1, 0, rc_stat)
        # rc_pvalue = 2*min(rc_pvalue, 1-rc_pvalue)

    # calculate statistic for T-test
    if not zero_mean:
        ttest_stat = (bias1 - bias2)/np.sqrt(var1/n1 + var2/n2)
        dof_num = (var1 / n1 + var2 / 2) ** 2
        dof_denom = (var1/n1) ** 2 / (n1 - 1) + (var2/n2)**2 / (n2 - 1)
        dof = dof_num/dof_denom
        t_pvalue = 2*stdtr(dof, -np.abs(ttest_stat))  # todo: seems wrong

    # calculate statistic for F-Test
    ftest_stat = var1/var2
    f_pvalue = ncfdtr(n1-1, n2-1, 0, ftest_stat)
    f_pvalue = 2*min(f_pvalue, 1-f_pvalue)

    print("coefs:", coefs1, coefs2)
    if len(cond_set) != 0:
        print("rc pvalue:", rc_pvalue)
    print("biases:", bias1, bias2)
    print("t pvalue:", t_pvalue)
    print("variances:", var1, var2)
    print("f pvalue:", f_pvalue)
    # print(i, cond_set)
    # if len(cond_set) != 0:
    #     print(f_pvalue, t_pvalue, rc_pvalue)
    # else:
    #     print(f_pvalue, t_pvalue)

    if len(cond_set) != 0 and not zero_mean:
        reject = f_pvalue<alpha/3 or rc_pvalue<alpha/3 or t_pvalue<alpha/3
    elif len(cond_set) == 0:
        reject = f_pvalue<alpha/2 or t_pvalue<alpha/2
    elif not zero_mean:
        reject = f_pvalue<alpha/2 or t_pvalue<alpha/2
    else:
        reject = f_pvalue<alpha
    print("Not invariant" if reject else "invariant")

    return dict(
        # ttest_stat=ttest_stat,
        ftest_stat=ftest_stat,
        f_pvalue=f_pvalue,
        # t_pvalue=t_pvalue,
        reject=reject
    )


