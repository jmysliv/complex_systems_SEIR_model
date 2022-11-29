import numpy as np
from adao import adaoBuilder
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIRECTORY_PATH = "../output"

labels = ["Susceptible","Exposed","Infected","Fatalities","Recovered"]
baseline = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/baseline.csv')
model_alpha = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/model_alpha.csv')

for label in labels:
    background = baseline[label].to_numpy()
    # add noise
    noise = np.random.normal(0,10,background.shape[0])
    background = background + noise
    observation = model_alpha[label].to_numpy()

    operator = np.identity(observation.shape[0])
    t = np.arange(observation.shape[0])

    case = adaoBuilder.New()
    case.set( 'AlgorithmParameters', Algorithm = '3DVAR' )
    case.set( 'Background',          Vector = background )
    case.set( 'Observation',         Vector = observation )
    case.set( 'ObservationOperator', Matrix = operator )
    case.execute()

    result = case.get('Analysis')[-1]

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
