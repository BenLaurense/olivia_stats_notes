import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""Helper functions for the Moments.ipynp notebook"""

RED = "#E24A33"
BLUE = "#348ABD"
GREY = "#777777"


def plot_dist(dist, xrange=None, ax=None):
    """Plots distribution"""
    if xrange is None:
        xrange = [0, 6]
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(f"pdf of distribution")
        ax.set_xlabel("x")
        ax.set_ylabel("pdf(x)")
        ax.set_xlim(xrange)
        ax.set_ylim([0, 1])

    xs = np.linspace(*xrange, 10 ** 3)
    ax.plot(xs, dist.pdf(xs), color=RED)
    return ax


def plot_dist_with_normal_approx(distribution, xrange=None, ax=None):
    """Plots distribution and its normal approximation"""
    if xrange is None:
        xrange = [0, 6]
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(f"pdf of distribution with normal approximation")
        ax.set_xlabel("x")
        ax.set_ylabel("pdf(x)")
        ax.set_xlim(xrange)
        ax.set_ylim([0, 1])

    mean, variance = distribution.mean(), distribution.var()
    xs = np.linspace(*xrange, 10 ** 3)
    ax.plot(xs, distribution.pdf(xs), color=RED, label="Distribution")
    ax.plot([mean, mean], [0, distribution.pdf(mean)], color=GREY, label="True mean")

    normal_approx = norm(mean, variance)
    ax.plot(xs, normal_approx.pdf(xs), color=BLUE, label="Normal Approximation")
    ax.legend()
    return ax
