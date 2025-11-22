import numpy as np


def ar_sampling(f, g, sample_g, m, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    samples = []
    while len(samples) < n:
        x = sample_g(1)[0]
        u = rng.uniform()
        if u < f(x) / (m * g(x)):
            samples.append(x)
    return np.array(samples)
