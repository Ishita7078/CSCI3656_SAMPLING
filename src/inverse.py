import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


def inverse_transform_analytic(f_inv, n, rng=None):
    #inverse transform sampling when inverse CDF is known
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(size=n)
    return f_inv(u)


def inverse_transform_numeric(pdf, x_min, x_max, n, grid_size=2000, rng=None):
    #mumerical inverse transform sampling for arbitrary unnormalized PDF
    if rng is None:
        rng = np.random.default_rng()

    xs = np.linspace(x_min, x_max, grid_size)
    pdf_vals = pdf(xs)

    #build CDF
    cdf_vals = cumulative_trapezoid(pdf_vals, xs, initial=0)
    cdf_vals /= cdf_vals[-1]

    #build inverse CDF
    F_inv = interp1d(cdf_vals, xs, bounds_error=False,
                     fill_value=(x_min, x_max))

    u = rng.uniform(size=n)
    return F_inv(u)
