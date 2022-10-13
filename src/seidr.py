from operator import mod
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class ModelResultsNotCalculatedError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("you need to call the calculate method first",*args)

class SEIDR:
    def __init__(self,N, beta, gamma, mi, lambda_, epsilon, alpha, no_of_days) -> None:
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.mi = mi
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.alpha = alpha

        self.no_of_days = no_of_days
        self.t = np.linspace(0, no_of_days, no_of_days+1)

        self.results = None

    def _deriv(self, initial_conditions, t):
        S, E, I, D, R = initial_conditions
        N = (S+E+I+R)
        dSdt = (-self.beta * S * I / N) -  self.mi*S +self.lambda_
        dEdt = self.beta * S * I / N - (self.mi+self.epsilon)*E
        dIdt = self.epsilon*E - (self.gamma+self.mi+self.alpha)*I
        dDdt = self.alpha*I
        dRdt = self.gamma * I - self.mi*R
        return dSdt, dEdt, dIdt, dDdt, dRdt

    def calculate(self, initial_conditions):
        self.results = odeint(self._deriv, initial_conditions, self.t)


    def plot(self):
        if self.results is None:
            raise ModelResultsNotCalculatedError()

        S, E, I, D, R = self.results.T

        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(self.t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(self.t, E/1000, 'orange', alpha=0.5, lw=2, label='Exposed')
        ax.plot(self.t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
        ax.plot(self.t, D/1000, 'black', alpha=0.5, lw=2, label='Fatalities')
        ax.plot(self.t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
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
    S0 = N - I0 - R0 - D0 - E0
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
    t = 160  # number of days to compute

    model = SEIDR( N, beta, gamma, mi, lambda_, epsilon, alpha, t)
    
    model.calculate(y0)

    model.plot()

    # Plot the data on three separate curves for S(t), I(t) and R(t)


