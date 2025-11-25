"""
Standalone demonstration of classic sampling methods (Part I).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from inverse import inverse_sampling
from acceptreject import ar_sampling
from mcmc import mcmc_sampling
from cluster import create_clusters, cluster_sampling


#mixture Gaussian PDF
def mixture_pdf(x):
    return 0.3 * norm.pdf(x, -2, 0.8) + 0.7 * norm.pdf(x, 2, 0.5)


proposal = lambda x: norm.pdf(x, 0, 4)
sample_proposal = lambda n: norm.rvs(0, 4, size=n)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    #inverse transform
    samples_inv = inverse_sampling(mixture_pdf, -6, 6, n=5000, rng=rng)
    xs = np.linspace(-6, 6, 500)

    plt.hist(samples_inv, bins=40, density=True, alpha=0.5)
    plt.plot(xs, mixture_pdf(xs))
    plt.title("Inverse Transform Sampling")
    plt.show()

    #accept–reject
    M = 4
    samples_rej = ar_sampling(mixture_pdf, proposal, sample_proposal, M, n=3000, rng=rng)
    plt.hist(samples_rej, bins=40, density=True, alpha=0.5)
    plt.plot(xs, mixture_pdf(xs))
    plt.title("Accept–Reject Sampling")
    plt.show()

    #mcmc
    samples_mcmc = mcmc_sampling(mixture_pdf, x0=0.0, n_steps=20000, proposal_std=0.5, rng=rng)
    plt.hist(samples_mcmc[5000:], bins=40, density=True, alpha=0.5)
    plt.plot(xs, mixture_pdf(xs))
    plt.title("MCMC Sampling")
    plt.show()

    #clusters
    params, pops = create_clusters(
        50,
        pdf=mixture_pdf,
        sample_fn=lambda n: inverse_sampling(mixture_pdf, -6, 6, n=n, rng=rng),
        rng=rng
    )

    idx = cluster_sampling(n_clusters=len(params), k=10, rng=rng)
    sampled_clusters = params[idx]
    plt.hist(sampled_clusters, bins=30)
    plt.title("Cluster Sampling")
    plt.show()
