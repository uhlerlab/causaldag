import numpy as np


def coin(p, size=1):
    return np.random.binomial(1, p, size=size)


def directed_erdos(n, s):
    pass


__all__ = ['directed_erdos']




