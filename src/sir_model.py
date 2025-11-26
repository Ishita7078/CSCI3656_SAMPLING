import numpy as np
from scipy.integrate import solve_ivp

def sir_rhs(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def solve_sir(beta, gamma, I0=0.01, t_span=(0,160), n_points=800):
    S0 = 1 - I0
    R0 = 0.0
    y0 = [S0, I0, R0]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        fun=lambda t, y: sir_rhs(t, y, beta, gamma),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45"
    )

    S, I, R = sol.y
    return sol.t, S, I, R

def sir_summary_stats(t, I, R):
    idx = np.argmax(I)
    peak_I = I[idx]
    t_peak = t[idx]
    final_R = R[-1]
    return peak_I, t_peak, final_R
