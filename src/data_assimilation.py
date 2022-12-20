import numpy as np
from adao import adaoBuilder
import pandas as pd
import matplotlib.pyplot as plt


LABELS = ["Susceptible","Exposed","Infected","Fatalities","Recovered"]

def apply_noise(model, deviation):
    noisy_model = {}
    for label in LABELS:
        serie = model[label].to_numpy()
        noise = np.random.normal(0,deviation,serie.shape[0])
        noisy_model[label] = serie + noise
    return noisy_model


def assimilate(noisy_baseline, surogate, baseline_var, surogate_var):
    results = {}
    for label in LABELS:
        background = noisy_baseline[label]
        
        observation = surogate[label].to_numpy()

        operator = np.identity(observation.shape[0])
        t = np.arange(observation.shape[0])

        case = adaoBuilder.New()
        case.set( 'AlgorithmParameters', Algorithm = '3DVAR' )
        case.set( 'Background',          Vector = background )
        case.set( 'BackgroundError',     ScalarSparseMatrix = baseline_var)       
        case.set( 'Observation',         Vector = observation )
        case.set( 'ObservationError',     ScalarSparseMatrix = surogate_var)       
        case.set( 'ObservationOperator', Matrix = operator )
        case.execute()

        result = case.get('Analysis')[-1]
        results[label] = result
    
    return tuple(results.values())


OUTPUT_DIRECTORY_PATH = "../output"


if __name__=="__main__":

    baseline = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/baseline.csv')
    surogate = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/surogate.csv')

    noisy_baseline = apply_noise(baseline, deviation=100)

    SEIDR = assimilate(noisy_baseline, surogate=surogate, baseline_var=1, surogate_var=10)
    # print(type(SEIDR[0]))
    

    for label, result in zip(LABELS, SEIDR):        
        background = noisy_baseline[label]

        observation = surogate[label].to_numpy()
        t = np.arange(observation.shape[0])

        # plot
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(t, baseline[label], 'black', alpha=0.5, lw=2, label=f'baseline')
        ax.plot(t, background, 'green', alpha=0.5, lw=2, label=f'noisy baseline')
        ax.plot(t, observation, 'red', alpha=0.5, lw=2, label=f'model')
        ax.plot(t, result, 'blue', alpha=0.5, lw=2, label=f'assimilation')
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
        plt.savefig(f"{OUTPUT_DIRECTORY_PATH}/assimilation/{label}.png")
        plt.close()
