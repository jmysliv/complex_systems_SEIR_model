import numpy as np
from adao import adaoBuilder
import pandas as pd
import matplotlib.pyplot as plt


LABELS = ["Susceptible","Exposed","Infected","Fatalities","Recovered"]

def assimilate(baseline, deviation, surogate, baseline_var, surogate_var):
    results = {}
    for label in LABELS:
        background = baseline[label].to_numpy()
        # add noise
        noise = np.random.normal(0,deviation,background.shape[0])
        background = background + noise
        observation = surogate[label].to_numpy()

        operator = np.identity(observation.shape[0])
        t = np.arange(observation.shape[0])

        case = adaoBuilder.New()
        case.set( 'AlgorithmParameters', Algorithm = '3DVAR' )
        case.set( 'Background',          Vector = background )
        case.set( 'BackgroundError',     ScalarSparseMatrix = baseline_var)       
        case.set( 'Observation',         Vector = observation )
        case.set( 'ObservationError',     ScalarSparseMatrix = baseline_var)       
        case.set( 'ObservationOperator', Matrix = operator )
        case.execute()

        result = case.get('Analysis')[-1]
        results[label] = result
    
    return tuple(results.values())


OUTPUT_DIRECTORY_PATH = "../output"


if __name__=="__main__":

    baseline = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/baseline.csv')
    model_alpha = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/model_alpha.csv')


    SEIDR = assimilate(baseline, deviation=10, surogate=model_alpha, baseline_var=1, surogate_var=1)
    # print(type(SEIDR[0]))
    

    for label, result in zip(LABELS, SEIDR):        
        background = baseline[label].to_numpy()

        background = background
        observation = model_alpha[label].to_numpy()
        t = np.arange(observation.shape[0])

        # plot
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(t, background, 'green', alpha=0.5, lw=2, label=f'baseline')
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
