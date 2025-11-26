import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from inverse import inverse_sampling
from acceptreject import ar_sampling
from mcmc import mcmc_sampling
from cluster import create_clusters, cluster_sampling

from sir_model import solve_sir, sir_summary_stats


def mixture_pdf(x):
    return 0.3 * norm.pdf(x, -2, 0.8) + 0.7 * norm.pdf(x, 2, 0.5)


proposal = lambda x: norm.pdf(x, 0, 4)
sample_proposal = lambda n: norm.rvs(0, 4, size=n)


plt.style.use("ggplot")



def normalize(arr):
    #handles negative or wide range samples
    a = np.array(arr)
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    return a

def map_to_beta_gamma(beta_raw, gamma_raw):
    #map samples into epidemiologically reasonable ranges

    beta = 0.3 + 0.4 * normalize(beta_raw)
    gamma = 0.05 + 0.10 * normalize(gamma_raw)
    return beta, gamma


def run_sir(betas, gammas):
    n = len(betas)
    peak_I = np.zeros(n)
    t_peak = np.zeros(n)
    final_R = np.zeros(n)

    for i in range(n):
        β = betas[i]
        γ = gammas[i]
        t, S, I, R = solve_sir(β, γ)
        peak_I[i], t_peak[i], final_R[i] = sir_summary_stats(t, I, R)

    return peak_I, t_peak, final_R


def plot_phase_plane(betas, gammas, title, filename, n_curves=50):
    idx = np.random.choice(len(betas), size=min(n_curves, len(betas)), replace=False)
    plt.figure(figsize=(7,6))
    for i in idx:
        β = betas[i]
        γ = gammas[i]
        t, S, I, R = solve_sir(β, γ)
        plt.plot(S, I, alpha=0.3)
    plt.xlabel("S")
    plt.ylabel("I")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_infected_curves(betas, gammas, title, filename, n_curves=50):
    idx = np.random.choice(len(betas), size=min(n_curves, len(betas)), replace=False)
    plt.figure(figsize=(8,5))
    for i in idx:
        β = betas[i]
        γ = gammas[i]
        t, S, I, R = solve_sir(β, γ)
        plt.plot(t, I, alpha=0.25)
    plt.xlabel("Time")
    plt.ylabel("I(t)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    xs = np.linspace(-6, 6, 500)

    samples_inv = inverse_sampling(mixture_pdf, -6, 6, n=5000, rng=rng)
    plt.hist(samples_inv, bins=40, density=True, alpha=0.5)
    plt.plot(xs, mixture_pdf(xs))
    plt.title("Inverse Transform Sampling")
    plt.tight_layout()
    plt.savefig("plots/inverse_sampling.png", dpi=200)
    plt.close()

    M = 4
    samples_rej = ar_sampling(mixture_pdf, proposal, sample_proposal, M, n=3000, rng=rng)
    plt.hist(samples_rej, bins=40, density=True, alpha=0.5)
    plt.plot(xs, mixture_pdf(xs))
    plt.title("Accept–Reject Sampling")
    plt.tight_layout()
    plt.savefig("plots/acceptreject_sampling.png", dpi=200)
    plt.close()

  
    samples_mcmc = mcmc_sampling(mixture_pdf, x0=0.0, n_steps=20000, proposal_std=0.5, rng=rng)
    plt.hist(samples_mcmc[5000:], bins=40, density=True, alpha=0.5)
    plt.plot(xs, mixture_pdf(xs))
    plt.title("MCMC Sampling")
    plt.tight_layout()
    plt.savefig("plots/mcmc_sampling.png", dpi=200)
    plt.close()

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
    plt.tight_layout()
    plt.savefig("plots/cluster_sampling.png", dpi=200)
    plt.close()

    n_sir = min(len(samples_inv), len(samples_rej))
    betas, gammas = map_to_beta_gamma(samples_inv[:n_sir], samples_rej[:n_sir])

    peak_I, t_peak, final_R = run_sir(betas, gammas)

    plt.hist(peak_I, bins=40)
    plt.title("Peak I — Inverse and AcceptReject")
    plt.tight_layout()
    plt.savefig("plots/sir_inverse_reject_peakI.png", dpi=200)
    plt.close()

    plot_phase_plane(betas, gammas, "Phase Plane Inverse and Reject","plots/sir_inverse_reject_phase.png")
    plot_infected_curves(betas, gammas,"I(t) Inverse and Reject","plots/sir_inverse_reject_I.png")

    #cluster
    betas_c, gammas_c = map_to_beta_gamma(sampled_clusters, sampled_clusters)
    plot_phase_plane(betas_c, gammas_c, "Phase Plane Cluster","plots/sir_cluster_phase.png")
    plot_infected_curves(betas_c, gammas_c,"I(t) Cluster", "plots/sir_cluster_I.png")


    #mcmc
    betas_m, gammas_m = map_to_beta_gamma(samples_mcmc[5000:5500], samples_mcmc[6000:6500])
    plot_phase_plane(betas_m, gammas_m, "Phase Plane MCMC","plots/sir_mcmc_phase.png")
    plot_infected_curves(betas_m, gammas_m, "I(t) MCMC", "plots/sir_mcmc_I.png")
