import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# The SIR model differential equations.
def SEIDR_deriv(y, t, N, beta, gamma, mi, lambda_, epsilon, alpha):
    S, E, I, D, R = y
    dSdt = (-beta * S * I / N) -  mi*S +lambda_
    dEdt = beta * S * I / N - (mi+epsilon)*E
    dIdt =  epsilon*E - (gamma+mi+alpha)*I
    dDdt = alpha*I
    dRdt = gamma * I - mi*R
    return dSdt, dEdt, dIdt, dDdt, dRdt

def SEIDR(initial_conditions, no_of_days, N, beta, gamma, mi, lambda_, epsilon, alpha):
    t = np.linspace(0, no_of_days, no_of_days+1)

    ret = odeint(SEIDR_deriv, initial_conditions, t, args=(N, beta, gamma, mi, lambda_, epsilon, alpha))
    return ret

def plot_SEIDR(SEIDR_results, no_of_days):
    S, E, I, D, R = SEIDR_results
    t = np.linspace(0, no_of_days, no_of_days+1)


    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E/1000, 'orange', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, D/1000, 'black', alpha=0.5, lw=2, label='Fatalities')
    ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

if __name__ == "__main__":
    # Total population, N.
    N = 1000
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, E0, D0, R0 = 0, 100, 0, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    beta, gamma = 0.5, 1./10 
    # A grid of time points (in days)
    # print(t)
    mi = 0.0002  # natural death rate
    alpha = 0.01  # fatality rate
    lambda_ = 0.0003*N  # natural birth rate
    epsilon = 0.2  # exposed to infectious
    # Initial conditions vector
    y0 = S0, E0, I0, D0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = SEIDR(y0, 160, N, beta, gamma, mi, lambda_, epsilon, alpha)

    plot_SEIDR(ret.T, 160)

    # Plot the data on three separate curves for S(t), I(t) and R(t)


