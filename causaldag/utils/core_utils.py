import itertools as itr
from numpy import abs
from typing import Iterable, Callable, Any


def ix_map_from_list(l):
    return {e: i for i, e in enumerate(l)}


def defdict2dict(defdict, keys):
    factory = defdict.default_factory
    d = {k: factory(v) for k, v in defdict.items()}
    for k in keys:
        if k not in d:
            d[k] = factory()
    return d


def powerset(s: Iterable, r_min=0, r_max=None) -> Iterable:
    if r_max is None: r_max = len(s)
    return map(set, itr.chain(*(itr.combinations(s, r) for r in range(r_min, r_max+1))))


def powerset_predicate(s: Iterable, predicate: Callable[[Any], bool]) -> Iterable:
    for set_size in range(len(s)+1):
        any_satisfy = False
        for subset in itr.combinations(s, set_size):
            if predicate(subset):
                any_satisfy = True
                yield set(subset)
        if not any_satisfy:
            break


def to_set(o) -> set:
    if not isinstance(o, set):
        try:
            return set(o)
        except TypeError:
            if o is None:
                return set()
            return {o}
    return o


def to_list(o):
    if not isinstance(o, list):
        try:
            return list(o)
        except TypeError:
            if o is None:
                return []
            return [o]
    return o


def is_symmetric(matrix, tol=1e-8):
    return (abs(matrix - matrix.T) <= tol).all()


if __name__ == '__main__':
    res = list(powerset_predicate(set(range(10)), lambda ss: len(ss) < 4))
