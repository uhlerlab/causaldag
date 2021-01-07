from typing import List, Iterable, Set, Dict, Hashable, Tuple, FrozenSet, Union
from warnings import warn
import inspect
Node = Hashable
DirectedEdge = Tuple[Node, Node]
UndirectedEdge = FrozenSet[Node]
BidirectedEdge = FrozenSet[Node]
NodeSet = Union[Hashable, Set[Hashable]]


def warn_untested():
    function_name = inspect.stack()[1][3]
    s = f"{function_name} still needs to be tested. If you intend to use this method, please submit a pull request."
    warn(s)

