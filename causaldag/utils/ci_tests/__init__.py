from .ci_tester import MemoizedCI_Tester, PlainCI_Tester, CI_Tester, PooledCI_Tester
from .partial_correlation_test import partial_correlation_test, partial_correlation_suffstat, partial_monte_carlo_correlation_suffstat, compute_partial_correlation
from .hsic import hsic_test_vector, hsic_test
from .fadcor import fadcor_test_vector, fadcor_test
from .kci import kci_test_vector, ki_test_vector, kci_test
from .oracle import dsep_test, msep_test


def get_ci_tester(
        samples,
        test="partial_correlation",
        memoize=False,
        **kwargs
):
    if test == "partial_correlation":
        ci_test = partial_correlation_test
        suffstat = partial_correlation_suffstat(samples)
    elif test == "hsic":
        ci_test = hsic_test
        suffstat = samples
    elif test == "kci":
        ci_test = kci_test
        suffstat = samples
    elif test == "dsep":
        ci_test = dsep_test
        suffstat = samples
    elif test == "msep":
        ci_test = msep_test
        suffstat = samples
    else:
        raise ValueError()

    if memoize:
        return MemoizedCI_Tester(ci_test, suffstat, **kwargs)
