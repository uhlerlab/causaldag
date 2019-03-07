from .kci import kci_test_vector, ki_test_vector, kci_test, kci_invariance_test
from .hsic import hsic_test_vector, hsic_test, hsic_invariance_test
from .gauss_ci import gauss_ci_test

from typing import NewType, Callable, Dict, Any, Union, List
import numpy as np
CI_Test = NewType('CI_Test', Callable[[Any, Union[int, List[int]], Union[int, List[int]]], Dict])
InvarianceTest = NewType('InvarianceTest', Callable[[np.ndarray, np.ndarray, int], Dict])
