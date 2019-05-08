import pygam
import numpy as np


def residuals(samples, i, j, cond_set):
    cond_set = list(cond_set)
    g = pygam.GAM()
    g.fit(samples[:, cond_set], samples[:, i])
    residuals_i = g.deviance_residuals(samples[:, cond_set], samples[:, i])
    g.fit(samples[:, cond_set], samples[:, j])
    residuals_j = g.deviance_residuals(samples[:, cond_set], samples[:, j])

    return residuals_i, residuals_j




