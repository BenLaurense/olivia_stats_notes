import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, rv_continuous

"""Helper functions for the Central Limit Theorem.ipynp notebook
All helpers expect a type implementing the methods: mean, var, rvs"""

BINS = 20
EDGECOLOUR = "black"


# Calculation helpers
def compute_sample_mean(distribution, n: int, num_samples: int = 10 ** 3) -> np.ndarray:
    sample_means = np.mean(distribution.rvs((n, num_samples)), axis=0)
    return sample_means


def get_normal_approx(distribution, n: int) -> rv_continuous:
    mean = distribution.mean()
    variance = distribution.var()
    # mean, variance = distribution.stats(moments='mv')
    scale = np.sqrt(variance)
    approx = norm(loc=mean, scale=np.sqrt(scale / n))
    return approx


# Plotting helpers
def distribution_histogram(distribution, num_samples: int = 10 ** 3, xrange=None, ax=None):
    """Plots the pdf and a histogram (with num_samples samples)
    of a specified distribution"""
    if xrange is None:
        xrange = [0, 6]
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(f"Histogram and pdf of distribution")
        ax.set_xlim(xrange)
        ax.set_ylim([0, 1])

    # Histogram
    samples = distribution.rvs(num_samples)
    ax.hist(samples, density=True, bins=BINS, edgecolor=EDGECOLOUR)

    # Plot
    xs = np.linspace(xrange[0], xrange[1], num_samples)
    ax.plot(xs, distribution.pdf(xs))
    return ax


def sample_mean_histogram(distribution, ns: list[int], num_samples: int = 10 ** 3, xrange=None, ax=None):
    """For each n in ns, plots histogram of samples from the distribution (X_1+...+X_n)/n
    where the X_i~distribution are iid"""
    if xrange is None:
        xrange = [0, 6]
    if ax is None:
        fig, ax = plt.subplots(len(ns), 1)
        fig.tight_layout(rect=(0, 0.05, 1, 0.9))
        title = ", ".join(map(lambda x: str(x), ns))
        fig.suptitle(f"Histograms for means of {title} copies of distribution")

    for idx in range(len(ns)):
        ax[idx].hist(compute_sample_mean(distribution, ns[idx], num_samples), density=True, bins=BINS, edgecolor=EDGECOLOUR)
        ax[idx].set_xlim(xrange)
        # ax[idx].set_ylim([0, 2])
        ax[idx].set_title(f"n={ns[idx]}")
    return ax


def sample_mean_histogram_with_approx(distribution, ns: list[int], num_samples: int = 10 ** 3, xrange=None, ax=None):
    """Same as sample_mean_histogram, except also plots the normal approximation
    given by the Central Limit Theorem"""
    if xrange is None:
        xrange = [0, 6]
    if ax is None:
        fig, ax = plt.subplots(len(ns), 1)
        fig.tight_layout(rect=(0, 0.05, 1, 0.9))
        title = ", ".join(map(lambda x: str(x), ns))
        fig.suptitle(f"Histograms for means of {title} copies of distribution")

    xs = np.linspace(xrange[0], xrange[1], num_samples)
    for idx in range(len(ns)):
        n = ns[idx]
        ax[idx].hist(compute_sample_mean(distribution, n, num_samples), density=True, bins=BINS, edgecolor=EDGECOLOUR)
        normal_approx = get_normal_approx(distribution, n)
        vals = normal_approx.pdf(xs)
        ax[idx].plot(xs, vals)
        ax[idx].set_xlim(xrange)
        ax[idx].set_ylim([0, max(1, max(vals) + 0.1)])
        ax[idx].set_title(f"n={n}")
    return ax


# Trace plots
def sample_mean_trace_plot(distribution, num_traces: int = 1, num_samples: int = 10, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
        fig.tight_layout(rect=(0, 0.05, 1, 0.9))
        fig.suptitle(f"Sample mean trace plot ({num_traces} traces)")

    true_mean = distribution.mean()
    ax.plot([0, num_samples - 1], [true_mean, true_mean], label="Distribution mean")

    # Put trajectory label on only one
    samples = distribution.rvs(num_samples)
    partial_sums = [sum(samples[:i]) / i for i in range(1, num_samples + 1)]
    if num_samples > 1:
        label = "Trajectories"
    else:
        label = "Trajectories"
    ax.plot(range(num_samples), partial_sums, label=label, color="teal")
    max_val = max(max(partial_sums), -min(partial_sums))

    # Vectorise
    for n in range(2, num_traces + 1):
        samples = distribution.rvs(num_samples)
        partial_sums = [sum(samples[:i]) / i for i in range(1, num_samples + 1)]
        ax.plot(range(num_samples), partial_sums, color="teal")
        max_val = max(max_val, max(partial_sums), -min(partial_sums))

    ax.set_ylim([-max_val - 0.5, max_val + 0.5])
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Sample average")
    ax.legend()
    return ax
