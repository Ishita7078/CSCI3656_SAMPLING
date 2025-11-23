import numpy as np


def mcmc_sampling(f, x0, n_steps, proposal_std=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    xs = np.empty(n_steps)
    x = x0
    fx = f(x)

    for i in range(n_steps):
        x_prop = x + proposal_std * rng.normal()
        fx_prop = f(x_prop)

        alpha = min(1.0, fx_prop / fx)
        if rng.uniform() < alpha:
            x = x_prop
            fx = fx_prop

        xs[i] = x

    return xs
