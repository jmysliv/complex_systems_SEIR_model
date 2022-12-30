import numpy as np
from adao import adaoBuilder
import pandas as pd
import matplotlib.pyplot as plt


LABELS = ["Susceptible","Exposed","Infected","Fatalities","Recovered"]

def apply_noise(model, deviation):
    noisy_model = model.copy()
    for label in LABELS:
        serie = model[label].to_numpy()
        noise = np.random.normal(0,deviation,serie.shape[0])
        noisy_model[label] += noise
    return noisy_model


def assimilate(noisy_baseline, surogate_params):
    case = adaoBuilder.New()
    case.set( 'AlgorithmParameters', Algorithm = '3DVAR' )
    case.set( 'Background',          Vector = surogate_params )
    case.set( 'Observation',         VectorSerie = noisy_baseline )
    case.set( 'ObservationOperator', OneFunction = True, Script = 'data_assimilation_funcs.py')
    case.execute()

    result = case.get('Analysis')[-1]
    return result


OUTPUT_DIRECTORY_PATH = "../output"


if __name__=="__main__":
    from data_assimilation_funcs import seidr_from_params

    baseline = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/baseline.csv', usecols=[i for i in range(1,6)])
    baseline = baseline[:80]
    surogate = [  # The best params I could get anything sensible with
        0.9, 0.45, 0.3, 0.15, 0.3, 0.02
    ]
    noisy_baseline = apply_noise(baseline, deviation=10).T  # The biggest noise I could get anything sensible with
    noisy_baseline.T.to_csv(f'{OUTPUT_DIRECTORY_PATH}/noisy_baseline.csv')
    noisy_baseline_pd = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/noisy_baseline.csv', usecols=[i for i in range(1,6)])
    noisy_baseline_pd.plot()
    plt.savefig(f'{OUTPUT_DIRECTORY_PATH}/noisy_baseline.png')


    model = seidr_from_params(surogate, name=f'assimilation/before_assimilation', no_of_days=400)
    model.plot()

    x = assimilate(noisy_baseline, surogate_params=surogate)
    
    model = seidr_from_params(x, name=f'assimilation/after_assimilation', no_of_days=400) 
    model.plot()
    print(x)
