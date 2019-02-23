import itertools as itr


def ix_map_from_list(l):
    return {e: i for i, e in enumerate(l)}


def defdict2dict(defdict, keys):
    factory = defdict.default_factory
    d = {k: factory(v) for k, v in defdict.items()}
    for k in keys:
        if k not in d:
            d[k] = factory()
    return d


def powerset(s):
    return itr.chain(*(itr.combinations(s, r) for r in range(len(s)+1)))


def to_set(o):
    if not isinstance(o, set):
        try:
            o = set(o)
        except TypeError:
            o = {o}
    return o

