import numpy as np


def create_clusters(n_clusters, pdf, sample_fn, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    params = sample_fn(n_clusters)
    pops = rng.integers(100, 5000, size=n_clusters)
    return params, pops


def cluster_sampling(n_clusters, k, rng=None):
    #randomly select k clusters
    if rng is None: 
        rng = np.random.default_rng()
    return rng.choice(n_clusters, size=k, replace=False)
