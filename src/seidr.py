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
                name: str, 
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

        self.name = name
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
        # immunity loss rate
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
        plt.savefig(f"{OUTPUT_DIRECTORY_PATH}/{self.name}.png")
        plt.close()

    def save_results(self):
        if self.results is None:
            raise ModelResultsNotCalculatedError()

        S, E, I, D, R = self.results.T
        results = zip( S, E, I, D, R)
        df = pd.DataFrame(results, columns=["Susceptible", "Exposed", "Infected", "Fatalities", "Recovered"])
        df.to_csv(f"{OUTPUT_DIRECTORY_PATH}/{self.name}.csv")

    def get_final_fatalities(self):
        if self.results is None:
            raise ModelResultsNotCalculatedError()

        _, _, _, D, _ = self.results.T
        return D[-1]


def compare_models(baseline: SEIDR, model: SEIDR):
    S1, E1, I1, D1, R1 = baseline.results.T
    S2, E2, I2, D2, R2 = model.results.T
    t = baseline.t

    labels = ['Susceptible', 'Exposed', 'Infected', 'Fatalities', 'Recovered with immunity']
    baseline_results = [S1, E1, I1, D1, R1]
    model_results = [S2, E2, I2, D2, R2]

    for label, baseline_result, model_result in zip(labels, baseline_results, model_results):
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(t, baseline_result, 'green', alpha=0.5, lw=2, label=f'{baseline.name}')
        ax.plot(t, model_result, 'red', alpha=0.5, lw=2, label=f'{model.name}')
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Population')
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        ax.set_title(label)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.savefig(f"{OUTPUT_DIRECTORY_PATH}/{model.name}_compare_{label}.png")
        plt.close()


if __name__ == "__main__":
    # baseline
    baseline = SEIDR(name="baseline") 
    baseline.calculate()
    baseline.save_results()
    baseline.plot()

    '''
    INSENSITIVE PARAMETER CHANGES
    '''
    # model beta 2
    model_beta_2 = SEIDR(name="model_beta_2", beta_values=[0.75, 1.0, 0.25]) 
    model_beta_2.calculate()
    model_beta_2.save_results()
    model_beta_2.plot()

    compare_models(baseline, model_beta_2)

    # model beta 1
    model_beta_1 = SEIDR(name="model_beta_1", beta_values=[0.1, 0.5, 0.25]) 
    model_beta_1.calculate()
    model_beta_1.save_results()
    model_beta_1.plot()

    compare_models(baseline, model_beta_1)

    # model mi
    model_mi = SEIDR(name="model_mi", mi=0.001) 
    model_mi.calculate()
    model_mi.save_results()
    model_mi.plot()

    compare_models(baseline, model_mi)

    '''
    SENSITIVE PARAMETER CHANGES
    '''
    # model gamma
    model_gamma = SEIDR(name="model_gamma", gamma=0.13) 
    model_gamma.calculate()
    model_gamma.save_results()
    model_gamma.plot()

    compare_models(baseline, model_gamma)

    # model alpha
    model_alpha = SEIDR(name="model_alpha", alpha=0.015) 
    model_alpha.calculate()
    model_alpha.save_results()
    model_alpha.plot()

    compare_models(baseline, model_alpha)

    # model beta3
    model_beta3 = SEIDR(name="model_beta3", beta_values=[0.75, 0.5, 0.3]) 
    model_beta3.calculate()
    model_beta3.save_results()
    model_beta3.plot()

    compare_models(baseline, model_beta3)

    # model delta
    model_delta = SEIDR(name="model_delta", delta=0.015) 
    model_delta.calculate()
    model_delta.save_results()
    model_delta.plot()

    compare_models(baseline, model_delta)
