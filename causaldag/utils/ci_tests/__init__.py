from .kci import kci_test_vector, ki_test_vector
from .gauss_ci import gauss_ci_test

from typing import NewType, Callable, Dict, Any
CI_Test = NewType('CI_Test', Callable[[Any, int, int, ...], Dict])
