from operator import mod
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIRECTORY_PATH = "../output"

class ModelResultsNotCalculatedError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("you need to call the calculate method first",*args)

class SEIDR:
    def __init__(self, 
                S = 9900,
                E = 100, 
                I = 0, 
                D = 0, 
                R = 0, 
                beta_bins = [20, 50],
                beta_values = [0.75, 0.5, 0.25],
                gamma = 0.1, 
                mi = 0.0002, 
                lambda_ = 0.0003, 
                epsilon = 0.2, 
                alpha = 0.01, 
                delta = 0.01,
                no_of_days = 400):

        # Susceptible
        self.S0 = S
        # exposed
        self.E0 = E
        # infected
        self.I0 = I
        # dead
        self.D0 = D
        # recovered
        self.R0 = R
        # bins for days when given contact rates ends
        self.beta_bins = beta_bins
        # array of contact rates values
        self.beta_values = beta_values
        # mean recovery rate, (in 1/days).
        self.gamma = gamma
        # natural death rate
        self.mi = mi
        # natural birth rate
        self.lambda_ = lambda_
        # exposed to infectious
        self.epsilon = epsilon
        # fatality rate
        self.alpha = alpha
        # possibility of losing immunity
        self.delta = delta

        self.no_of_days = no_of_days
        self.t = np.linspace(0, no_of_days, no_of_days+1)
        self.results = None


    def _deriv(self, y, t):
        current_beta = self.beta_values[np.digitize(t, self.beta_bins)]
        S, E, I, _, R = y
        N = (S+E+I+R)
        dSdt = (-current_beta * S * I / N) -  self.mi * S + self.lambda_ * N + self.delta * R
        dEdt = current_beta * S * I / N - (self.mi + self.epsilon) * E
        dIdt = self.epsilon * E - (self.gamma + self.mi + self.alpha) * I
        dDdt = self.alpha * I
        dRdt = self.gamma * I - self.mi * R - self.delta * R
        return dSdt, dEdt, dIdt, dDdt, dRdt

    def calculate(self):
        y0 = self.S0, self.E0, self.I0, self.D0, self.R0
        self.results = odeint(self._deriv, y0, self.t)


    def plot(self):
        if self.results is None:
            raise ModelResultsNotCalculatedError()

        S, E, I, D, R = self.results.T

        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(self.t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(self.t, E, 'orange', alpha=0.5, lw=2, label='Exposed')
        ax.plot(self.t, I, 'r', alpha=0.5, lw=2, label='Infected')
        ax.plot(self.t, D, 'black', alpha=0.5, lw=2, label='Fatalities')
        ax.plot(self.t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Population')
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.savefig(f"{OUTPUT_DIRECTORY_PATH}/S{self.S0}_E{self.E0}_I{self.I0}_D{self.D0}_R{self.R0}_a{self.alpha}_g{self.gamma}_m{self.mi}_l{self.lambda_}_e{self.epsilon}_d{self.delta}.png")
        plt.close()

    def save_results(self):
        if self.results is None:
            raise ModelResultsNotCalculatedError()

        S, E, I, D, R = self.results.T
        results = zip( S, E, I, D, R)
        df = pd.DataFrame(results, columns=["Susceptible", "Exposed", "Infected", "Fatalities", "Recovered"])
        df.to_csv(f"{OUTPUT_DIRECTORY_PATH}/S{self.S0}_E{self.E0}_I{self.I0}_D{self.D0}_R{self.R0}_a{self.alpha}_g{self.gamma}_m{self.mi}_l{self.lambda_}_e{self.epsilon}_d{self.delta}.csv")



if __name__ == "__main__":
    model = SEIDR() 
    model.calculate()
    model.save_results()
    model.plot()
